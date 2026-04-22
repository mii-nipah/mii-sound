//! Frontend-side TTS engine: spawns and supervises a long-lived
//! `mii-sound tts-worker` child process. Forwards requests over stdin/stdout
//! using the wire protocol, propagates client cancellation via SIGUSR1
//! (so the worker keeps its loaded model and we don't pay the cold start),
//! and unloads the worker (kill + wait) after `--holds` of inactivity to
//! reclaim VRAM.

use crate::proto::{self, OP_TTS, Request, ST_GENERATION_FAILED, ST_MODEL_MISSING};
use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::AsyncReadExt;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, Notify};

pub enum TtsResult {
    Ok(Bytes),
    ModelMissing(String),
    Cancelled,
    Failed(String),
}

pub struct TtsEngine {
    inner: Arc<Inner>,
}

struct Inner {
    model_dir: PathBuf,
    cpu: bool,
    ttl: Duration,
    state: Mutex<Option<RunningWorker>>,
}

struct RunningWorker {
    child: Child,
    pid: u32,
    stdin: ChildStdin,
    stdout: ChildStdout,
    last_used: Instant,
}

impl TtsEngine {
    pub fn new(model_dir: PathBuf, ttl: Duration, cpu: bool) -> Self {
        let inner = Arc::new(Inner {
            model_dir,
            cpu,
            ttl,
            state: Mutex::new(None),
        });
        spawn_eviction_task(inner.clone());
        TtsEngine { inner }
    }

    pub async fn generate(
        &self,
        json: Bytes,
        audio: Option<Bytes>,
        cancel: Arc<Notify>,
    ) -> TtsResult {
        let mut guard = self.inner.state.lock().await;

        // Spawn worker on demand.
        if guard.is_none() {
            match spawn_worker(&self.inner.model_dir, self.inner.cpu).await {
                Ok(w) => *guard = Some(w),
                Err(e) => return TtsResult::ModelMissing(e.to_string()),
            }
        }
        let worker = guard.as_mut().expect("worker just spawned");

        // Detect crash before sending: if child died, drop and try once more.
        if let Ok(Some(status)) = worker.child.try_wait() {
            log::warn!("tts worker exited unexpectedly ({status}); respawning");
            *guard = None;
            // Recurse once via re-spawn.
            drop(guard);
            return Box::pin(retry_after_crash(self.inner.clone(), json, audio, cancel)).await;
        }

        let req = Request {
            op: OP_TTS,
            json,
            audio,
        };
        if let Err(e) = proto::write_request(&mut worker.stdin, &req).await {
            log::warn!("tts worker write failed: {e}; dropping worker");
            *guard = None;
            return TtsResult::Failed(format!("worker pipe error: {e}"));
        }

        let pid = worker.pid;
        let response = tokio::select! {
            r = proto::read_response(&mut worker.stdout) => r,
            _ = cancel.notified() => {
                send_cancel_signal(pid);
                // Still wait for the worker to surface its Cancelled response
                // (it will, once VoxCPM exits the diffusion loop).
                proto::read_response(&mut worker.stdout).await
            }
        };

        match response {
            Ok(resp) => {
                worker.last_used = Instant::now();
                if resp.status == proto::ST_OK {
                    TtsResult::Ok(resp.payload)
                } else if resp.status == ST_MODEL_MISSING {
                    TtsResult::ModelMissing(payload_str(&resp.payload))
                } else if resp.status == ST_GENERATION_FAILED
                    && resp.payload.as_ref() == b"cancelled"
                {
                    TtsResult::Cancelled
                } else {
                    TtsResult::Failed(payload_str(&resp.payload))
                }
            }
            Err(e) => {
                log::warn!("tts worker read failed: {e}; dropping worker");
                *guard = None;
                TtsResult::Failed(format!("worker pipe error: {e}"))
            }
        }
    }
}

async fn retry_after_crash(
    inner: Arc<Inner>,
    json: Bytes,
    audio: Option<Bytes>,
    cancel: Arc<Notify>,
) -> TtsResult {
    let engine = TtsEngine { inner };
    engine.generate(json, audio, cancel).await
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
