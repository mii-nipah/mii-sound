//! CLI definitions (clap derive).

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Parser)]
#[command(
    name = "mii-sound",
    version,
    about = "Composable sound generation utility"
)]
pub struct Cli {
    /// Connect to a remote server (host:port). When unset, the local UDS is used.
    #[arg(long, global = true)]
    pub url: Option<String>,

    /// Custom UDS socket path (server: bind here, client: connect here).
    #[arg(long, global = true)]
    pub socket: Option<PathBuf>,

    /// Check whether the server is reachable. Exit 0 if up, 1 if not.
    #[arg(long)]
    pub status: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    /// Run as the server (resource holder + request handler).
    Serve(ServeArgs),
    /// Text-to-speech client.
    Tts(TtsArgs),
    /// Internal: long-lived TTS worker process. Speaks the wire protocol over
    /// stdin/stdout. Spawned automatically by `serve`; not intended to be run
    /// by hand.
    #[command(name = "tts-worker", hide = true)]
    TtsWorker(TtsWorkerArgs),
}

#[derive(Debug, Clone, Parser)]
pub struct TtsWorkerArgs {
    #[arg(long)]
    pub tts_dir: PathBuf,

    #[arg(long)]
    pub cpu: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct ServeArgs {
    /// Path to the VoxCPM2 model directory (config.json + tokenizer.json + model.safetensors + audiovae.pth).
    #[arg(long)]
    pub tts_dir: Option<PathBuf>,

    /// Force CPU backend (default: wgpu).
    #[arg(long)]
    pub cpu: bool,

    /// How long to keep an idle resource loaded before unloading. Examples: 30s, 10m, 1h, 3d.
    #[arg(long, default_value = "10m", value_parser = parse_duration)]
    pub holds: Duration,

    /// Listen on a TCP port instead of UDS. Pair with the $TOKEN env var.
    #[arg(long)]
    pub network: Option<u16>,

    /// Forward all requests to a remote `mii-http`-hosted mii-sound server.
    /// Local clients keep using the UDS / TCP socket as usual; resources are
    /// loaded on the remote machine. Token comes from `$MII_SOUND_TOKEN`.
    /// Format: `host[:port]`, optionally with an `http://`/`https://` scheme.
    #[arg(long, value_name = "URL")]
    pub relay: Option<String>,

    /// Suppress per-request and lifecycle logs (errors are still printed).
    #[arg(long)]
    pub quiet: bool,
}

#[derive(Debug, Clone, Parser)]
pub struct TtsArgs {
    /// Write the resulting audio to this file instead of stdout.
    #[arg(long)]
    pub out: Option<PathBuf>,

    /// Pass the request JSON inline instead of via stdin.
    #[arg(long)]
    pub json: Option<String>,

    /// Enable voice cloning (the request JSON must include a `reference`).
    #[arg(long)]
    pub voice_clone: bool,

    /// Classifier-free guidance / adherence to the prompt.
    #[arg(long = "cfg", visible_aliases = ["adherence"], short = 'a', default_value_t = 2.0)]
    pub cfg: f32,

    /// Number of diffusion steps.
    #[arg(long, short = 's', default_value_t = 10)]
    pub steps: u32,

    /// Stream the audio out as it is generated, instead of waiting for the
    /// whole utterance. Output is a 16-bit PCM mono WAV; on stdout the RIFF
    /// and `data` chunk sizes are written as `0xFFFFFFFF` (the streaming
    /// sentinel most decoders accept), while `--out` finalizes them once the
    /// stream completes.
    #[arg(long)]
    pub stream: bool,
}

fn parse_duration(s: &str) -> Result<Duration, String> {
    humantime::parse_duration(s).map_err(|e| format!("invalid duration `{s}`: {e}"))
}
