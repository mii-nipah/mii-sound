//! Server core: bind a local socket (interprocess) or TCP, per-connection
//! handler, dispatch.

pub mod relay;
pub mod tts;

use crate::cli::ServeArgs;
use crate::proto::{
    self, KIND_END, KIND_ERROR, OP_STATUS, OP_TTS, OP_TTS_STREAM, Request, Response,
    ST_BAD_REQUEST, ST_GENERATION_FAILED, ST_MODEL_MISSING, ST_OK, TtsRequest,
};
use crate::transport;
use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use interprocess::local_socket::ListenerOptions;
use interprocess::local_socket::tokio::Stream as IpcStream;
use interprocess::local_socket::traits::tokio::{
    Listener as IpcListenerTrait, Stream as IpcStreamTrait,
};
use relay::Relay;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use tts::{TtsEngine, TtsResult};

struct ServerCtx {
    tts: Option<TtsEngine>,
    relay: Option<Arc<Relay>>,
    expected_token: Option<String>,
}

pub async fn run(args: ServeArgs, socket_override: Option<PathBuf>) -> Result<()> {
    if args.relay.is_some() && args.tts_dir.is_some() {
        log::warn!(
            "--relay and --tts-dir are both set; relay mode will forward all requests and the local model dir is ignored"
        );
    }

    let relay = match args.relay.as_ref() {
        Some(url) => {
            let r = Relay::new(url).context("setting up relay client")?;
            log::info!("relay mode: forwarding requests to {}", r.upstream());
            Some(Arc::new(r))
        }
        None => None,
    };

    let tts_engine = if relay.is_some() {
        None
    } else {
        args.tts_dir
            .as_ref()
            .map(|dir| TtsEngine::new(dir.clone(), args.holds, args.cpu))
    };

    if relay.is_none() {
        if let Some(dir) = args.tts_dir.as_ref() {
            log::info!(
                "tts engine ready (backend={}, model_dir={}, holds={})",
                if args.cpu { "cpu" } else { "wgpu" },
                dir.display(),
                humantime::format_duration(args.holds),
            );
        } else {
            log::info!("no --tts-dir given; tts requests will be rejected");
        }
    }

    let expected_token = if args.network.is_some() {
        Some(transport::token_from_env().unwrap_or_default())
    } else {
        None
    };

    let ctx = Arc::new(ServerCtx {
        tts: tts_engine,
        relay,
        expected_token,
    });

    if let Some(port) = args.network {
        let bind = format!("0.0.0.0:{port}");
        let listener = TcpListener::bind(&bind)
            .await
            .with_context(|| format!("binding {bind}"))?;
        log::info!("serving on tcp://{bind}");
        tokio::select! {
            r = accept_loop_tcp(listener, ctx) => r,
            _ = shutdown_signal() => {
                log::info!("shutdown signal received");
                Ok(())
            }
        }
    } else {
        let name = transport::resolve_name(socket_override.as_deref())?;
        // Best-effort parent dir creation when binding to a filesystem path.
        if let Some(path) = socket_override.as_deref()
            && let Some(parent) = path.parent()
        {
            tokio::fs::create_dir_all(parent).await.ok();
        }

        let mut opts = ListenerOptions::new().name(name);
        // Try to overwrite a stale socket file from a previous run.
        opts = opts.try_overwrite(true);
        // Restrict perms when applicable.
        #[cfg(unix)]
        {
            use interprocess::os::unix::local_socket::ListenerOptionsExt;
            opts = opts.mode(0o600);
        }

        let listener = opts.create_tokio().context("binding local socket")?;
        log::info!("serving on local socket");

        tokio::select! {
            r = accept_loop_local(listener, ctx) => r,
            _ = shutdown_signal() => {
                log::info!("shutdown signal received");
                Ok(())
            }
        }
    }
}

#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn accept_loop_local(
    listener: interprocess::local_socket::tokio::Listener,
    ctx: Arc<ServerCtx>,
) -> Result<()> {
    loop {
        let stream = listener.accept().await?;
        let ctx = ctx.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_local(stream, ctx).await {
                log::warn!("connection error: {e}");
            }
        });
    }
}

async fn accept_loop_tcp(listener: TcpListener, ctx: Arc<ServerCtx>) -> Result<()> {
    loop {
        let (stream, peer) = listener.accept().await?;
        let ctx = ctx.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_tcp(stream, ctx).await {
                log::warn!("connection from {peer} error: {e}");
            }
        });
    }
}

async fn handle_local(stream: IpcStream, ctx: Arc<ServerCtx>) -> Result<()> {
    let (read, write) = stream.split();
    handle(read, write, ctx, false).await
}

async fn handle_tcp(stream: TcpStream, ctx: Arc<ServerCtx>) -> Result<()> {
    let (read, write) = stream.into_split();
    handle(read, write, ctx, true).await
}

async fn handle<R, W>(
    mut read: R,
    mut write: W,
    ctx: Arc<ServerCtx>,
    expect_token: bool,
) -> Result<()>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send,
{
    if expect_token {
        let token = proto::read_token(&mut read)
            .await
            .context("reading token")?;
        let expected = ctx.expected_token.clone().unwrap_or_default();
        if token != expected {
            let resp = Response {
                status: ST_BAD_REQUEST,
                payload: Bytes::from_static(b"invalid token"),
            };
            let _ = proto::write_response(&mut write, &resp).await;
            return Err(anyhow!("invalid token"));
        }
    }

    let req = proto::read_request(&mut read)
        .await
        .context("reading request")?;

    if req.op == OP_TTS_STREAM {
        return handle_stream(req, &ctx, read, &mut write).await;
    }

    if let Some(relay) = ctx.relay.clone() {
        let response = dispatch_relay(req, relay, read).await;
        proto::write_response(&mut write, &response).await?;
        return Ok(());
    }

    let response = dispatch(req, &ctx, read).await;
    proto::write_response(&mut write, &response).await?;
    Ok(())
}

async fn dispatch<R>(req: Request, ctx: &Arc<ServerCtx>, mut read: R) -> Response
where
    R: AsyncRead + Unpin + Send + 'static,
{
    match req.op {
        OP_STATUS => Response {
            status: ST_OK,
            payload: Bytes::new(),
        },
        OP_TTS => {
            let Some(engine) = ctx.tts.as_ref() else {
                return error_response(ST_MODEL_MISSING, "server was started without --tts-dir");
            };
            // Parse just for logging; forward raw json bytes to the worker.
            let parsed: TtsRequest = match serde_json::from_slice(&req.json) {
                Ok(v) => v,
                Err(e) => return error_response(ST_BAD_REQUEST, format!("invalid tts json: {e}")),
            };

            let preview: String = parsed.text.chars().take(60).collect();
            log::info!(
                "tts request received: cfg={} steps={} chars={} preview={:?}",
                parsed.cfg,
                parsed.steps,
                parsed.text.chars().count(),
                preview
            );
            let started = std::time::Instant::now();

            let cancel = Arc::new(Notify::new());
            let watcher_cancel = cancel.clone();
            let watcher = tokio::spawn(async move {
                let mut buf = [0u8; 16];
                loop {
                    match read.read(&mut buf).await {
                        Ok(0) | Err(_) => {
                            watcher_cancel.notify_one();
                            return;
                        }
                        Ok(_) => {}
                    }
                }
            });

            let result = engine.generate(req.json, req.audio, cancel).await;
            watcher.abort();

            match result {
                TtsResult::Ok(payload) => {
                    log::info!(
                        "tts request finished in {:.2?} ({} bytes wav)",
                        started.elapsed(),
                        payload.len()
                    );
                    Response {
                        status: ST_OK,
                        payload,
                    }
                }
                TtsResult::ModelMissing(msg) => error_response(ST_MODEL_MISSING, msg),
                TtsResult::Cancelled => {
                    log::info!("tts request cancelled after {:.2?}", started.elapsed());
                    error_response(ST_GENERATION_FAILED, "cancelled")
                }
                TtsResult::Failed(msg) => error_response(ST_GENERATION_FAILED, msg),
            }
        }
        other => error_response(ST_BAD_REQUEST, format!("unknown op {other}")),
    }
}

fn error_response(status: u8, msg: impl Into<String>) -> Response {
    let s = msg.into();
    log::warn!("request failed (status={status}): {s}");
    Response {
        status,
        payload: Bytes::from(s.into_bytes()),
    }
}

async fn handle_stream<R, W>(
    req: Request,
    ctx: &Arc<ServerCtx>,
    mut read: R,
    write: &mut W,
) -> Result<()>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send,
{
    use crate::proto::StreamFrame;

    if let Some(relay) = ctx.relay.clone() {
        return handle_stream_relay(req, relay, read, write).await;
    }

    let Some(engine) = ctx.tts.as_ref() else {
        log::warn!("stream request rejected: server started without --tts-dir");
        let frame = StreamFrame {
            kind: KIND_ERROR,
            payload: proto::error_payload(ST_MODEL_MISSING, "server was started without --tts-dir"),
        };
        proto::write_stream_frame(write, &frame).await?;
        return Ok(());
    };

    let parsed: TtsRequest = match serde_json::from_slice(&req.json) {
        Ok(v) => v,
        Err(e) => {
            let frame = StreamFrame {
                kind: KIND_ERROR,
                payload: proto::error_payload(ST_BAD_REQUEST, &format!("invalid tts json: {e}")),
            };
            proto::write_stream_frame(write, &frame).await?;
            return Ok(());
        }
    };

    let preview: String = parsed.text.chars().take(60).collect();
    log::info!(
        "tts stream request received: cfg={} steps={} chars={} preview={:?}",
        parsed.cfg,
        parsed.steps,
        parsed.text.chars().count(),
        preview
    );
    let started = std::time::Instant::now();

    let cancel = Arc::new(Notify::new());
    let watcher_cancel = cancel.clone();
    let watcher = tokio::spawn(async move {
        let mut buf = [0u8; 16];
        loop {
            match read.read(&mut buf).await {
                Ok(0) | Err(_) => {
                    watcher_cancel.notify_one();
                    return;
                }
                Ok(_) => {}
            }
        }
    });

    let mut stream = engine.generate_stream(req.json, req.audio, cancel).await;
    let mut final_kind: Option<u8> = None;
    while let Some(frame) = stream.rx.recv().await {
        let kind = frame.kind;
        proto::write_stream_frame(write, &frame).await?;
        if matches!(kind, KIND_END | KIND_ERROR) {
            final_kind = Some(kind);
            break;
        }
    }
    watcher.abort();

    match final_kind {
        Some(KIND_END) => {
            log::info!("tts stream finished in {:.2?}", started.elapsed());
        }
        Some(KIND_ERROR) => {
            log::info!(
                "tts stream ended with error after {:.2?}",
                started.elapsed()
            );
        }
        _ => {
            log::warn!(
                "tts stream channel closed without terminal frame after {:.2?}",
                started.elapsed()
            );
            // Provide a terminal error so the client doesn't hang.
            let frame = StreamFrame {
                kind: KIND_ERROR,
                payload: proto::error_payload(
                    ST_GENERATION_FAILED,
                    "stream ended without terminal frame",
                ),
            };
            proto::write_stream_frame(write, &frame).await?;
        }
    }
    Ok(())
}

// --- Relay-mode dispatch ---------------------------------------------------

async fn dispatch_relay<R>(req: Request, relay: Arc<Relay>, mut read: R) -> Response
where
    R: AsyncRead + Unpin + Send + 'static,
{
    match req.op {
        OP_STATUS => {
            if relay.status().await {
                Response {
                    status: ST_OK,
                    payload: Bytes::new(),
                }
            } else {
                error_response(ST_BAD_REQUEST, "relay upstream not running")
            }
        }
        OP_TTS => {
            let parsed: TtsRequest = match serde_json::from_slice(&req.json) {
                Ok(v) => v,
                Err(e) => return error_response(ST_BAD_REQUEST, format!("invalid tts json: {e}")),
            };
            let preview: String = parsed.text.chars().take(60).collect();
            log::info!(
                "relay tts request: cfg={} steps={} chars={} preview={:?}",
                parsed.cfg,
                parsed.steps,
                parsed.text.chars().count(),
                preview
            );
            let started = std::time::Instant::now();

            // Watch the client connection so we abort the upstream call on
            // disconnect.
            let cancel = Arc::new(Notify::new());
            let watcher_cancel = cancel.clone();
            let watcher = tokio::spawn(async move {
                let mut buf = [0u8; 16];
                loop {
                    match read.read(&mut buf).await {
                        Ok(0) | Err(_) => {
                            watcher_cancel.notify_one();
                            return;
                        }
                        Ok(_) => {}
                    }
                }
            });

            let request = relay.tts(&parsed, req.audio);
            let result = tokio::select! {
                r = request => Some(r),
                _ = cancel.notified() => None,
            };
            watcher.abort();

            match result {
                None => {
                    log::info!(
                        "relay tts request cancelled by client after {:.2?}",
                        started.elapsed()
                    );
                    error_response(ST_GENERATION_FAILED, "cancelled")
                }
                Some(Ok(payload)) => {
                    log::info!(
                        "relay tts request finished in {:.2?} ({} bytes)",
                        started.elapsed(),
                        payload.len()
                    );
                    Response {
                        status: ST_OK,
                        payload,
                    }
                }
                Some(Err(e)) => error_response(e.status, e.message),
            }
        }
        other => error_response(ST_BAD_REQUEST, format!("unknown op {other}")),
    }
}

async fn handle_stream_relay<R, W>(
    req: Request,
    relay: Arc<Relay>,
    mut read: R,
    write: &mut W,
) -> Result<()>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send,
{
    use crate::proto::StreamFrame;

    let parsed: TtsRequest = match serde_json::from_slice(&req.json) {
        Ok(v) => v,
        Err(e) => {
            let frame = StreamFrame {
                kind: KIND_ERROR,
                payload: proto::error_payload(ST_BAD_REQUEST, &format!("invalid tts json: {e}")),
            };
            proto::write_stream_frame(write, &frame).await?;
            return Ok(());
        }
    };

    let preview: String = parsed.text.chars().take(60).collect();
    log::info!(
        "relay tts stream request: cfg={} steps={} chars={} preview={:?}",
        parsed.cfg,
        parsed.steps,
        parsed.text.chars().count(),
        preview
    );
    let started = std::time::Instant::now();

    // Spawn a watcher to detect client disconnect. We can't propagate a true
    // cancel into reqwest (no shared cancel token); instead we abort the
    // forwarder task, which drops the upstream response and ends the request.
    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::channel::<StreamFrame>(8);
    let relay_for_task = relay.clone();
    let parsed_for_task = parsed.clone();
    let audio_for_task = req.audio.clone();
    let forward = tokio::spawn(async move {
        relay_for_task
            .tts_stream(&parsed_for_task, audio_for_task, frame_tx)
            .await;
    });

    let mut disconnect_buf = [0u8; 16];
    let mut final_kind: Option<u8> = None;

    loop {
        tokio::select! {
            biased;
            r = read.read(&mut disconnect_buf) => {
                match r {
                    Ok(0) | Err(_) => {
                        log::info!(
                            "relay tts stream cancelled by client after {:.2?}",
                            started.elapsed()
                        );
                        forward.abort();
                        return Ok(());
                    }
                    Ok(_) => continue,
                }
            }
            maybe_frame = frame_rx.recv() => {
                match maybe_frame {
                    Some(frame) => {
                        let kind = frame.kind;
                        proto::write_stream_frame(write, &frame).await?;
                        if matches!(kind, KIND_END | KIND_ERROR) {
                            final_kind = Some(kind);
                            break;
                        }
                    }
                    None => break,
                }
            }
        }
    }

    let _ = forward.await;

    match final_kind {
        Some(KIND_END) => {
            log::info!("relay tts stream finished in {:.2?}", started.elapsed());
        }
        Some(KIND_ERROR) => {
            log::info!(
                "relay tts stream ended with error after {:.2?}",
                started.elapsed()
            );
        }
        _ => {
            log::warn!(
                "relay tts stream channel closed without terminal frame after {:.2?}",
                started.elapsed()
            );
            let frame = StreamFrame {
                kind: KIND_ERROR,
                payload: proto::error_payload(
                    ST_GENERATION_FAILED,
                    "relay stream ended without terminal frame",
                ),
            };
            proto::write_stream_frame(write, &frame).await?;
        }
    }
    Ok(())
}
