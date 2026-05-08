//! Frontend-side TTS engine: spawns and supervises a long-lived
//! `mii-sound tts-worker` child process. Forwards requests over stdin/stdout
//! using the wire protocol, propagates client cancellation via SIGUSR1
//! (so the worker keeps its loaded model and we don't pay the cold start),
//! and unloads the worker (kill + wait) after `--holds` of inactivity to
//! reclaim VRAM.

use crate::proto::{
    self, KIND_END, KIND_ERROR, OP_TTS_STREAM, Request, ST_GENERATION_FAILED, ST_MODEL_MISSING,
    StreamFrame, TtsRequest,
};
use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, Notify, oneshot};

/// Default grace window if a caller passes `Duration::ZERO`. Matches the
/// behavior described in `specs.md` and the `--batch-window` default.
const DEFAULT_BATCH_GRACE: Duration = Duration::from_millis(300);

pub enum TtsResult {
    Ok(Bytes),
    ModelMissing(String),
    Cancelled,
    Failed(String),
}

/// Server-side stream relay: holds the in-flight worker pipe lock for the
/// duration of the generation and yields raw [`StreamFrame`]s as the worker
/// produces them. Drop the receiver to abandon further frames (the connection
/// watcher will drive cancellation independently).
pub struct TtsStream {
    pub rx: tokio::sync::mpsc::Receiver<StreamFrame>,
}

pub struct TtsEngine {
    inner: Arc<Inner>,
}

struct Inner {
    model_dir: PathBuf,
    cpu: bool,
    ttl: Duration,
    parallel: usize,
    batch_window: Duration,
    state: Mutex<Option<RunningWorker>>,
    queue: Mutex<VecDeque<PendingItem>>,
    queue_notify: Notify,
}

struct RunningWorker {
    child: Child,
    pid: u32,
    stdin: ChildStdin,
    stdout: ChildStdout,
    last_used: Instant,
}

/// One waiting non-streaming request. Items are pulled by the dispatcher,
/// grouped by [`BatchKey`], and dispatched as a single batched forward pass.
struct PendingItem {
    json: Bytes,
    audio: Option<Bytes>,
    key: BatchKey,
    cancel: Arc<Notify>,
    response_tx: oneshot::Sender<TtsResult>,
}

/// Items only batch together if they share these generation parameters,
/// because `voxcpm-rs::BatchBuilder::run` takes a single `GenerateOptions`
/// for the whole batch.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct BatchKey {
    cfg_bits: u32,
    steps: u32,
}

impl BatchKey {
    fn from_request(req: &TtsRequest) -> Self {
        BatchKey {
            cfg_bits: req.cfg.to_bits(),
            steps: req.steps,
        }
    }
}

impl TtsEngine {
    pub fn new(
        model_dir: PathBuf,
        ttl: Duration,
        cpu: bool,
        parallel: usize,
        batch_window: Duration,
    ) -> Self {
        let parallel = parallel.max(1);
        let batch_window = if batch_window.is_zero() {
            DEFAULT_BATCH_GRACE
        } else {
            batch_window
        };
        let inner = Arc::new(Inner {
            model_dir,
            cpu,
            ttl,
            parallel,
            batch_window,
            state: Mutex::new(None),
            queue: Mutex::new(VecDeque::new()),
            queue_notify: Notify::new(),
        });
        spawn_eviction_task(inner.clone());
        spawn_dispatcher(inner.clone());
        if parallel > 1 {
            log::info!(
                "tts batching enabled (parallel={parallel}, window={})",
                humantime::format_duration(batch_window)
            );
        }
        TtsEngine { inner }
    }

    /// Submit one TTS request to the batching scheduler. The returned future
    /// resolves once the worker has produced a response (or the request was
    /// cancelled by `cancel`).
    pub async fn generate(
        &self,
        json: Bytes,
        audio: Option<Bytes>,
        cancel: Arc<Notify>,
    ) -> TtsResult {
        // Parse cfg/steps for batch grouping. The full json bytes are still
        // forwarded verbatim to the worker, so any other fields stay intact.
        let parsed: TtsRequest = match serde_json::from_slice(&json) {
            Ok(p) => p,
            Err(e) => return TtsResult::Failed(format!("invalid tts json: {e}")),
        };
        let key = BatchKey::from_request(&parsed);

        let (tx, rx) = oneshot::channel();
        let item = PendingItem {
            json,
            audio,
            key,
            cancel,
            response_tx: tx,
        };
        {
            let mut q = self.inner.queue.lock().await;
            q.push_back(item);
        }
        self.inner.queue_notify.notify_one();

        match rx.await {
            Ok(r) => r,
            Err(_) => TtsResult::Failed("dispatcher dropped request".to_string()),
        }
    }

    /// Streaming counterpart to [`generate`]. Spawns a relay task that holds
    /// the worker pipe lock for the duration of the generation and forwards
    /// every frame from the worker into the returned receiver verbatim.
    /// On worker spawn failure we synthesize a single `KIND_ERROR` frame so
    /// the caller can treat the channel uniformly.
    pub async fn generate_stream(
        &self,
        json: Bytes,
        audio: Option<Bytes>,
        cancel: Arc<Notify>,
    ) -> TtsStream {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamFrame>(8);
        let inner = self.inner.clone();
        tokio::spawn(stream_relay(inner, json, audio, cancel, tx));
        TtsStream { rx }
    }
}

async fn stream_relay(
    inner: Arc<Inner>,
    json: Bytes,
    audio: Option<Bytes>,
    cancel: Arc<Notify>,
    tx: tokio::sync::mpsc::Sender<StreamFrame>,
) {
    let mut guard = inner.state.lock().await;

    if guard.is_none() {
        match spawn_worker(&inner.model_dir, inner.cpu).await {
            Ok(w) => *guard = Some(w),
            Err(e) => {
                let frame = StreamFrame {
                    kind: KIND_ERROR,
                    payload: proto::error_payload(ST_MODEL_MISSING, &e.to_string()),
                };
                let _ = tx.send(frame).await;
                return;
            }
        }
    }
    let worker = guard.as_mut().expect("worker just spawned");

    if let Ok(Some(status)) = worker.child.try_wait() {
        log::warn!("tts worker exited unexpectedly ({status}); will respawn on next request");
        *guard = None;
        let frame = StreamFrame {
            kind: KIND_ERROR,
            payload: proto::error_payload(
                ST_GENERATION_FAILED,
                &format!("worker exited before request ({status})"),
            ),
        };
        let _ = tx.send(frame).await;
        return;
    }

    let req = Request {
        op: OP_TTS_STREAM,
        json,
        audio,
    };
    if let Err(e) = proto::write_request(&mut worker.stdin, &req).await {
        log::warn!("tts worker write failed: {e}; dropping worker");
        *guard = None;
        let frame = StreamFrame {
            kind: KIND_ERROR,
            payload: proto::error_payload(ST_GENERATION_FAILED, &format!("worker pipe error: {e}")),
        };
        let _ = tx.send(frame).await;
        return;
    }

    let pid = worker.pid;
    let mut cancel_signaled = false;
    loop {
        let frame_result = tokio::select! {
            r = proto::read_stream_frame(&mut worker.stdout) => r,
            _ = cancel.notified(), if !cancel_signaled => {
                send_cancel_signal(pid);
                cancel_signaled = true;
                continue;
            }
        };
        let frame = match frame_result {
            Ok(f) => f,
            Err(e) => {
                log::warn!("tts worker stream read failed: {e}; dropping worker");
                *guard = None;
                let err_frame = StreamFrame {
                    kind: KIND_ERROR,
                    payload: proto::error_payload(
                        ST_GENERATION_FAILED,
                        &format!("worker pipe error: {e}"),
                    ),
                };
                let _ = tx.send(err_frame).await;
                return;
            }
        };
        let terminal = matches!(frame.kind, KIND_END | KIND_ERROR);
        // If the receiver is gone, just drain until the worker reports
        // terminal status (so the worker pipe stays in sync for the next
        // request). Cancellation should already be in flight.
        let _ = tx.send(frame).await;
        if terminal {
            worker.last_used = Instant::now();
            return;
        }
    }
}

// --- Batching dispatcher ---------------------------------------------------

fn spawn_dispatcher(inner: Arc<Inner>) {
    tokio::spawn(dispatcher_loop(inner));
}

async fn dispatcher_loop(inner: Arc<Inner>) {
    loop {
        // Wait for at least one item.
        let first = loop {
            {
                let mut q = inner.queue.lock().await;
                if let Some(it) = q.pop_front() {
                    break it;
                }
            }
            inner.queue_notify.notified().await;
        };

        let key = first.key;
        let max = inner.parallel;
        let mut batch: Vec<PendingItem> = Vec::with_capacity(max);
        batch.push(first);

        if max > 1 {
            // Opportunistic: drain matching items already queued.
            drain_matching(&inner, &mut batch, key, max).await;

            // Grace window: wait up to batch_window for more matching items.
            if batch.len() < max {
                let deadline = tokio::time::Instant::now() + inner.batch_window;
                while batch.len() < max {
                    let now = tokio::time::Instant::now();
                    if now >= deadline {
                        break;
                    }
                    let remaining = deadline - now;
                    tokio::select! {
                        _ = tokio::time::sleep(remaining) => break,
                        _ = inner.queue_notify.notified() => {
                            drain_matching(&inner, &mut batch, key, max).await;
                        }
                    }
                }
            }
        }

        if batch.len() > 1 {
            log::info!("tts dispatcher batched {} requests", batch.len());
        }

        dispatch_batch(&inner, batch).await;
    }
}

/// Pull up to `max - batch.len()` items matching `key` from the queue into
/// `batch`. Items with mismatched keys are left untouched at their position
/// (so they keep their FIFO order for the next dispatcher round).
async fn drain_matching(
    inner: &Arc<Inner>,
    batch: &mut Vec<PendingItem>,
    key: BatchKey,
    max: usize,
) {
    let mut q = inner.queue.lock().await;
    let mut i = 0;
    while batch.len() < max && i < q.len() {
        if q[i].key == key {
            batch.push(q.remove(i).expect("indexed item exists"));
        } else {
            i += 1;
        }
    }
}

async fn dispatch_batch(inner: &Arc<Inner>, batch: Vec<PendingItem>) {
    let mut guard = inner.state.lock().await;

    if guard.is_none() {
        match spawn_worker(&inner.model_dir, inner.cpu).await {
            Ok(w) => *guard = Some(w),
            Err(e) => {
                let msg = e.to_string();
                for item in batch {
                    let _ = item.response_tx.send(TtsResult::ModelMissing(msg.clone()));
                }
                return;
            }
        }
    }
    let worker = guard.as_mut().expect("worker just spawned");

    // Crash check: if the child died between requests, drop it and requeue
    // the items at the front so the next dispatcher iteration retries with a
    // fresh worker.
    if let Ok(Some(status)) = worker.child.try_wait() {
        log::warn!("tts worker exited unexpectedly ({status}); respawning");
        *guard = None;
        drop(guard);
        requeue_front(inner, batch).await;
        return;
    }

    let items: Vec<proto::BatchItem> = batch
        .iter()
        .map(|p| proto::BatchItem {
            json: p.json.clone(),
            audio: p.audio.clone(),
        })
        .collect();

    if let Err(e) = proto::write_batch_request(&mut worker.stdin, &items).await {
        log::warn!("tts worker batch write failed: {e}; dropping worker");
        *guard = None;
        let msg = format!("worker pipe error: {e}");
        for item in batch {
            let _ = item.response_tx.send(TtsResult::Failed(msg.clone()));
        }
        return;
    }

    let pid = worker.pid;
    let total = batch.len();
    let cancelled_count = Arc::new(AtomicUsize::new(0));
    let mut watchers = Vec::with_capacity(total);
    for item in &batch {
        let notify = item.cancel.clone();
        let count = cancelled_count.clone();
        let h = tokio::spawn(async move {
            notify.notified().await;
            let prev = count.fetch_add(1, Ordering::SeqCst);
            if prev + 1 == total {
                // Every client in this batch has dropped — voxcpm only
                // exposes batch-wide cancellation, so we can only safely
                // cancel when nobody is still listening.
                send_cancel_signal(pid);
            }
        });
        watchers.push(h);
    }

    let response = proto::read_batch_response(&mut worker.stdout).await;

    for h in watchers {
        h.abort();
    }

    match response {
        Ok(resps) => {
            worker.last_used = Instant::now();
            if resps.len() != total {
                log::warn!(
                    "tts worker returned {} responses for batch of {}",
                    resps.len(),
                    total
                );
            }
            for (item, resp) in batch.into_iter().zip(resps) {
                let result = if resp.status == proto::ST_OK {
                    TtsResult::Ok(resp.payload)
                } else if resp.status == ST_MODEL_MISSING {
                    TtsResult::ModelMissing(payload_str(&resp.payload))
                } else if resp.status == ST_GENERATION_FAILED
                    && resp.payload.as_ref() == b"cancelled"
                {
                    TtsResult::Cancelled
                } else {
                    TtsResult::Failed(payload_str(&resp.payload))
                };
                let _ = item.response_tx.send(result);
            }
        }
        Err(e) => {
            log::warn!("tts worker batch read failed: {e}; dropping worker");
            *guard = None;
            let msg = format!("worker pipe error: {e}");
            for item in batch {
                let _ = item.response_tx.send(TtsResult::Failed(msg.clone()));
            }
        }
    }
}

async fn requeue_front(inner: &Arc<Inner>, batch: Vec<PendingItem>) {
    let mut q = inner.queue.lock().await;
    // Reverse so the original head ends up at the front again.
    for item in batch.into_iter().rev() {
        q.push_front(item);
    }
    drop(q);
    inner.queue_notify.notify_one();
}

fn payload_str(b: &Bytes) -> String {
    String::from_utf8_lossy(b).into_owned()
}

#[cfg(unix)]
fn send_cancel_signal(pid: u32) {
    unsafe extern "C" {
        fn kill(pid: i32, sig: i32) -> i32;
    }
    const SIGUSR1: i32 = 10; // Linux value; matches every glibc target we ship to.
    unsafe {
        if kill(pid as i32, SIGUSR1) != 0 {
            log::warn!("failed to deliver SIGUSR1 to tts worker pid {pid}");
        } else {
            log::info!("forwarded cancellation to tts worker (pid={pid})");
        }
    }
}

#[cfg(not(unix))]
fn send_cancel_signal(_pid: u32) {
    log::warn!("cancellation forwarding not implemented on this platform");
}

async fn spawn_worker(model_dir: &PathBuf, cpu: bool) -> Result<RunningWorker> {
    let exe = std::env::current_exe().context("locating mii-sound binary")?;
    log::info!(
        "spawning tts worker (backend={}, model_dir={})",
        if cpu { "cpu" } else { "wgpu" },
        model_dir.display()
    );
    let mut cmd = Command::new(&exe);
    cmd.arg("tts-worker")
        .arg("--tts-dir")
        .arg(model_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    if cpu {
        cmd.arg("--cpu");
    }
    // Forward RUST_LOG so the worker's logs show up at the same verbosity.
    if std::env::var_os("RUST_LOG").is_none() {
        cmd.env("RUST_LOG", "info");
    }
    #[cfg(target_os = "linux")]
    {
        // Safety: pre_exec runs after fork, before exec; we only call
        // async-signal-safe libc functions here.
        unsafe {
            std::os::unix::process::CommandExt::pre_exec(cmd.as_std_mut(), || {
                unsafe extern "C" {
                    fn prctl(option: i32, arg2: u64, ...) -> i32;
                }
                const PR_SET_PDEATHSIG: i32 = 1;
                const SIGTERM: u64 = 15;
                // PR_SET_PDEATHSIG: kill the worker if the parent dies (e.g.
                // we segfault) so it doesn't outlive us.
                prctl(PR_SET_PDEATHSIG, SIGTERM);
                Ok(())
            });
        }
    }

    let mut child = cmd.spawn().context("spawning tts worker process")?;
    let pid = child
        .id()
        .ok_or_else(|| anyhow!("worker child has no pid"))?;
    let stdin = child.stdin.take().expect("piped stdin");
    let mut stdout = child.stdout.take().expect("piped stdout");

    // Wait for the ready byte before considering the worker usable.
    let ready_started = Instant::now();
    let mut ready = [0u8; 1];
    match stdout.read_exact(&mut ready).await {
        Ok(_) if ready[0] == 0 => {
            log::info!(
                "tts worker ready (pid={pid}, startup={:.2?})",
                ready_started.elapsed()
            );
        }
        Ok(_) => {
            return Err(anyhow!("worker sent unexpected ready byte: {}", ready[0]));
        }
        Err(e) => {
            // Drain the child to surface the real failure if it died early.
            let _ = child.kill().await;
            return Err(anyhow!("worker did not become ready: {e}"));
        }
    }

    Ok(RunningWorker {
        child,
        pid,
        stdin,
        stdout,
        last_used: Instant::now(),
    })
}

fn spawn_eviction_task(inner: Arc<Inner>) {
    if inner.ttl.is_zero() {
        return;
    }
    let tick = inner
        .ttl
        .min(Duration::from_secs(10))
        .max(Duration::from_secs(1));
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tick);
        interval.tick().await; // discard immediate tick
        loop {
            interval.tick().await;
            let mut guard = inner.state.lock().await;
            let should_evict = match guard.as_ref() {
                Some(w) => w.last_used.elapsed() >= inner.ttl,
                None => false,
            };
            if !should_evict {
                continue;
            }
            let Some(mut worker) = guard.take() else {
                continue;
            };
            log::info!(
                "unloading tts worker (idle for {})",
                humantime::format_duration(inner.ttl)
            );
            // Closing stdin lets the worker exit cleanly; if it hangs, kill.
            drop(worker.stdin);
            match tokio::time::timeout(Duration::from_secs(3), worker.child.wait()).await {
                Ok(Ok(status)) => {
                    log::info!("tts worker exited cleanly ({status})");
                }
                Ok(Err(e)) => log::warn!("waiting for tts worker failed: {e}"),
                Err(_) => {
                    log::warn!("tts worker did not exit in 3s; killing");
                    let _ = worker.child.kill().await;
                }
            }
        }
    });
}
