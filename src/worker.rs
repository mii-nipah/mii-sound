//! TTS worker process. Loads a single VoxCPM model and serves requests
//! framed with the same wire protocol used by the public socket. Talks
//! exclusively over stdin (requests) and stdout (responses); status logs go to
//! stderr.
//!
//! Cancellation is forwarded from the supervisor via `SIGUSR1`: the supervisor
//! sends the signal to the worker PID, the worker handler flips the current
//! request's [`CancelToken`], and `synthesize` returns `Cancelled` from the
//! diffusion loop. The worker process keeps running.

use crate::cli::TtsWorkerArgs;
use crate::proto::{
    self, OP_TTS, Request, Response, ST_BAD_REQUEST, ST_GENERATION_FAILED, ST_OK, TtsRequest,
};
use crate::synth::{self, Model, SynthError};
use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;
use tokio::io::{AsyncWriteExt, BufReader};
use voxcpm_rs::CancelToken;

pub async fn run(args: TtsWorkerArgs) -> Result<()> {
    log::info!(
        "tts worker starting (backend={}, model_dir={})",
        if args.cpu { "cpu" } else { "wgpu" },
        args.tts_dir.display()
    );
    let load_started = Instant::now();
    let dir = args.tts_dir.clone();
    let cpu = args.cpu;
    let model = tokio::task::spawn_blocking(move || synth::load(&dir, cpu))
        .await
        .map_err(|e| anyhow!("worker load task panicked: {e}"))??;
    log::info!("tts worker ready (loaded in {:.2?})", load_started.elapsed());
    let model = Arc::new(StdMutex::new(model));

    // Slot holding the in-flight request's cancel token so SIGUSR1 can flip
    // it without restarting the worker process.
    let current_cancel: Arc<StdMutex<Option<CancelToken>>> = Arc::new(StdMutex::new(None));

    install_cancel_handler(current_cancel.clone());

    // Tell the supervisor we're ready by writing a single zero byte (an
    // out-of-band sentinel). The frontend reads this byte before sending the
    // first real request so it can include the load time in startup metrics.
    let mut stdout = tokio::io::stdout();
    stdout.write_u8(0).await.context("writing ready byte")?;
    stdout.flush().await?;

    let mut stdin = BufReader::new(tokio::io::stdin());

    loop {
        let req = match proto::read_request(&mut stdin).await {
            Ok(r) => r,
            Err(e) => {
                log::info!("tts worker stdin closed: {e}");
                return Ok(());
            }
        };
        let resp =
            handle_request(model.clone(), current_cancel.clone(), req).await;
        if let Err(e) = proto::write_response(&mut stdout, &resp).await {
            log::warn!("tts worker stdout closed: {e}");
            return Ok(());
        }
    }
}

async fn handle_request(
    model: Arc<StdMutex<Model>>,
    current_cancel: Arc<StdMutex<Option<CancelToken>>>,
    req: Request,
) -> Response {
    if req.op != OP_TTS {
        return error_response(ST_BAD_REQUEST, format!("worker received non-tts op {}", req.op));
    }
    let parsed: TtsRequest = match serde_json::from_slice(&req.json) {
        Ok(v) => v,
        Err(e) => return error_response(ST_BAD_REQUEST, format!("invalid tts json: {e}")),
    };

    let cancel = CancelToken::new();
    *current_cancel.lock().expect("cancel slot poisoned") = Some(cancel.clone());

    let started = Instant::now();
    log::info!(
        "tts worker processing: cfg={} steps={} chars={}",
        parsed.cfg,
        parsed.steps,
        parsed.text.chars().count(),
    );

    let join = tokio::task::spawn_blocking(move || {
        let guard = model.lock().expect("model mutex poisoned");
        synth::synthesize(&*guard, parsed, req.audio, cancel)
    });
    let result = join.await;

    *current_cancel.lock().expect("cancel slot poisoned") = None;

    match result {
        Ok(Ok(payload)) => {
            log::info!(
                "tts worker finished in {:.2?} ({} bytes wav)",
                started.elapsed(),
                payload.len()
            );
            Response { status: ST_OK, payload }
        }
        Ok(Err(SynthError::Cancelled)) => {
            log::info!(
                "tts worker request cancelled after {:.2?}",
                started.elapsed()
            );
            error_response(ST_GENERATION_FAILED, "cancelled")
        }
        Ok(Err(SynthError::Other(msg))) => error_response(ST_GENERATION_FAILED, msg),
        Err(e) => error_response(ST_GENERATION_FAILED, format!("worker task panicked: {e}")),
    }
}

fn error_response(status: u8, msg: impl Into<String>) -> Response {
    let s = msg.into();
    log::warn!("tts worker error (status={status}): {s}");
    Response {
        status,
        payload: Bytes::from(s.into_bytes()),
    }
}

#[cfg(unix)]
fn install_cancel_handler(slot: Arc<StdMutex<Option<CancelToken>>>) {
    use tokio::signal::unix::{SignalKind, signal};
    tokio::spawn(async move {
        let mut sig = match signal(SignalKind::user_defined1()) {
            Ok(s) => s,
            Err(e) => {
                log::error!("failed to install SIGUSR1 handler: {e}");
                return;
            }
        };
        while sig.recv().await.is_some() {
            if let Some(token) = slot.lock().expect("cancel slot poisoned").as_ref() {
                token.cancel();
                log::info!("tts worker received cancel signal");
            }
        }
    });
}

#[cfg(not(unix))]
fn install_cancel_handler(_slot: Arc<StdMutex<Option<CancelToken>>>) {
    // TODO: implement on Windows (e.g. via a control pipe).
}
