//! Client side: connect to a local socket (interprocess, cross-platform) or
//! TCP, send a request, handle the response.

pub mod tts;

use crate::cli::Cli;
use crate::exit;
use crate::proto::{
    self, KIND_CHUNK, KIND_END, KIND_ERROR, KIND_HEADER, OP_STATUS, Request, Response,
};
use crate::transport;
use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;
use interprocess::local_socket::tokio::Stream as IpcStream;
use interprocess::local_socket::traits::tokio::Stream as IpcStreamTrait;
use std::path::Path;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

pub enum Conn {
    Local(IpcStream),
    Tcp(TcpStream),
}

pub async fn connect(cli: &Cli) -> Result<Conn> {
    if let Some(url) = &cli.url {
        let mut stream = TcpStream::connect(url)
            .await
            .with_context(|| format!("failed to connect to {url}"))?;
        let token = transport::token_from_env().unwrap_or_default();
        proto::write_token(&mut stream, &token).await?;
        Ok(Conn::Tcp(stream))
    } else {
        let name = transport::resolve_name(cli.socket.as_deref())?;
        let stream = IpcStream::connect(name)
            .await
            .context("failed to connect to local socket")?;
        Ok(Conn::Local(stream))
    }
}

pub async fn send_recv(conn: &mut Conn, req: &Request) -> Result<Response> {
    match conn {
        Conn::Local(s) => exchange(s, req).await,
        Conn::Tcp(s) => exchange(s, req).await,
    }
}

async fn exchange<S>(stream: &mut S, req: &Request) -> Result<Response>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    proto::write_request(stream, req).await?;
    proto::read_response(stream).await
}

pub async fn run_status(cli: &Cli) -> i32 {
    let mut conn = match connect(cli).await {
        Ok(c) => c,
        Err(_) => {
            println!("unreachable");
            return exit::SERVER_UNREACHABLE;
        }
    };
    let req = Request {
        op: OP_STATUS,
        json: Bytes::new(),
        audio: None,
    };
    match send_recv(&mut conn, &req).await {
        Ok(resp) if resp.status == proto::ST_OK => {
            println!("running");
            exit::SUCCESS
        }
        _ => {
            println!("unreachable");
            exit::SERVER_UNREACHABLE
        }
    }
}

pub fn status_to_exit(status: u8) -> i32 {
    match status {
        proto::ST_OK => exit::SUCCESS,
        proto::ST_MODEL_MISSING => exit::MODEL_NOT_FOUND,
        proto::ST_GENERATION_FAILED => exit::GENERATION_FAILED,
        proto::ST_BAD_REQUEST => exit::BAD_REQUEST,
        _ => exit::UNKNOWN,
    }
}

pub async fn read_all_stdin() -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut stdin = tokio::io::stdin();
    stdin.read_to_end(&mut buf).await?;
    Ok(buf)
}

pub fn split_json_and_rest(data: &[u8]) -> Result<(serde_json::Value, &[u8])> {
    let mut stream = serde_json::Deserializer::from_slice(data).into_iter::<serde_json::Value>();
    let value = stream
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty JSON input"))?
        .context("invalid JSON on stdin")?;
    let rest_start = stream.byte_offset();
    Ok((value, &data[rest_start..]))
}

pub async fn write_payload_out(out: Option<&std::path::Path>, payload: &[u8]) -> Result<()> {
    match out {
        Some(path) => tokio::fs::write(path, payload)
            .await
            .with_context(|| format!("writing {}", path.display())),
        None => {
            let mut stdout = tokio::io::stdout();
            stdout.write_all(payload).await?;
            stdout.flush().await?;
            Ok(())
        }
    }
}

pub fn fail_bad_request(msg: impl std::fmt::Display) -> ! {
    eprintln!("mii-sound: bad request: {msg}");
    std::process::exit(exit::BAD_REQUEST);
}

pub fn fail_unreachable(msg: impl std::fmt::Display) -> ! {
    eprintln!("mii-sound: server unreachable: {msg}");
    std::process::exit(exit::SERVER_UNREACHABLE);
}

pub fn fail_unknown(msg: impl std::fmt::Display) -> ! {
    eprintln!("mii-sound: error: {msg}");
    std::process::exit(exit::UNKNOWN);
}

pub fn handle_response_status(resp: &Response) {
    if resp.status != proto::ST_OK {
        let msg = std::str::from_utf8(&resp.payload).unwrap_or("<non-utf8 error>");
        if !msg.is_empty() {
            eprintln!("mii-sound: server error: {msg}");
        }
        std::process::exit(status_to_exit(resp.status));
    }
}

/// Run a streaming request: forwards the request, reads stream frames, and
/// writes a 16-bit PCM mono WAV to either stdout or `out` as samples arrive.
/// Returns Ok(()) on a clean stream end, or `Err((status, msg))` on a
/// terminal error frame / pipe failure (status mapped to a `proto::ST_*`).
pub async fn stream_request(
    conn: &mut Conn,
    req: &Request,
    out: Option<&Path>,
) -> std::result::Result<(), (u8, String)> {
    match conn {
        Conn::Local(s) => stream_exchange(s, req, out).await,
        Conn::Tcp(s) => stream_exchange(s, req, out).await,
    }
}

async fn stream_exchange<S>(
    stream: &mut S,
    req: &Request,
    out: Option<&Path>,
) -> std::result::Result<(), (u8, String)>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    proto::write_request(stream, req)
        .await
        .map_err(|e| (proto::ST_UNKNOWN, format!("write request: {e}")))?;

    let mut sink = match out {
        Some(path) => StreamSink::file(path)
            .await
            .map_err(|e| (proto::ST_UNKNOWN, e.to_string()))?,
        None => StreamSink::stdout(),
    };

    loop {
        let frame = proto::read_stream_frame(stream)
            .await
            .map_err(|e| (proto::ST_UNKNOWN, format!("read stream frame: {e}")))?;
        match frame.kind {
            KIND_HEADER => {
                let sr = proto::parse_header_payload(&frame.payload)
                    .map_err(|e| (proto::ST_UNKNOWN, e.to_string()))?;
                sink.write_header(sr)
                    .await
                    .map_err(|e| (proto::ST_UNKNOWN, e.to_string()))?;
            }
            KIND_CHUNK => {
                sink.write_samples(&frame.payload)
                    .await
                    .map_err(|e| (proto::ST_UNKNOWN, e.to_string()))?;
            }
            KIND_END => {
                sink.finalize()
                    .await
                    .map_err(|e| (proto::ST_UNKNOWN, e.to_string()))?;
                return Ok(());
            }
            KIND_ERROR => {
                let (status, msg) = proto::parse_error_payload(&frame.payload);
                return Err((status, msg));
            }
            other => {
                return Err((
                    proto::ST_UNKNOWN,
                    format!("unknown stream frame kind {other}"),
                ));
            }
        }
    }
}

/// 16-bit PCM mono WAV header (44 bytes). When `data_size == u32::MAX` the
/// chunk-size field also gets `u32::MAX`, which is the conventional "size
/// unknown / streaming" sentinel most decoders accept.
fn wav_header(sample_rate: u32, data_size: u32) -> [u8; 44] {
    let chunk_size: u32 = if data_size == u32::MAX {
        u32::MAX
    } else {
        36u32.saturating_add(data_size)
    };
    let byte_rate = sample_rate.saturating_mul(2); // 16-bit mono
    let block_align: u16 = 2;
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&chunk_size.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    h[22..24].copy_from_slice(&1u16.to_le_bytes()); // channels
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    h[32..34].copy_from_slice(&block_align.to_le_bytes());
    h[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits per sample
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&data_size.to_le_bytes());
    h
}

fn samples_f32_le_to_pcm16(payload: &[u8]) -> Result<Vec<u8>> {
    if !payload.len().is_multiple_of(4) {
        bail!("chunk payload not a multiple of 4 bytes");
    }
    let mut out = Vec::with_capacity(payload.len() / 2);
    for chunk in payload.chunks_exact(4) {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        let s = f32::from_le_bytes(buf).clamp(-1.0, 1.0);
        let v = (s * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

enum StreamSink {
    Stdout {
        out: tokio::io::Stdout,
        header_written: bool,
    },
    File {
        file: tokio::fs::File,
        path: std::path::PathBuf,
        bytes_written: u32,
        header_written: bool,
        sample_rate: u32,
    },
}

impl StreamSink {
    fn stdout() -> Self {
        Self::Stdout {
            out: tokio::io::stdout(),
            header_written: false,
        }
    }

    async fn file(path: &Path) -> Result<Self> {
        let file = tokio::fs::File::create(path)
            .await
            .with_context(|| format!("creating {}", path.display()))?;
        Ok(Self::File {
            file,
            path: path.to_path_buf(),
            bytes_written: 0,
            header_written: false,
            sample_rate: 0,
        })
    }

    async fn write_header(&mut self, sample_rate: u32) -> Result<()> {
        match self {
            Self::Stdout {
                out,
                header_written,
            } => {
                if *header_written {
                    return Err(anyhow!("duplicate stream header frame"));
                }
                let h = wav_header(sample_rate, u32::MAX);
                out.write_all(&h).await?;
                out.flush().await?;
                *header_written = true;
            }
            Self::File {
                file,
                header_written,
                sample_rate: sr_slot,
                ..
            } => {
                if *header_written {
                    return Err(anyhow!("duplicate stream header frame"));
                }
                // Write a placeholder header; finalize() rewrites with real sizes.
                let h = wav_header(sample_rate, 0);
                file.write_all(&h).await?;
                *header_written = true;
                *sr_slot = sample_rate;
            }
        }
        Ok(())
    }

    async fn write_samples(&mut self, payload: &[u8]) -> Result<()> {
        let pcm = samples_f32_le_to_pcm16(payload)?;
        match self {
            Self::Stdout {
                out,
                header_written,
            } => {
                if !*header_written {
                    return Err(anyhow!("chunk frame received before header frame"));
                }
                out.write_all(&pcm).await?;
                out.flush().await?;
            }
            Self::File {
                file,
                bytes_written,
                header_written,
                ..
            } => {
                if !*header_written {
                    return Err(anyhow!("chunk frame received before header frame"));
                }
                file.write_all(&pcm).await?;
                *bytes_written = bytes_written.saturating_add(pcm.len() as u32);
            }
        }
        Ok(())
    }

    async fn finalize(self) -> Result<()> {
        match self {
            Self::Stdout { mut out, .. } => {
                out.flush().await?;
            }
            Self::File {
                mut file,
                path,
                bytes_written,
                sample_rate,
                header_written,
            } => {
                if !header_written {
                    // We never received a header — leave the empty file as-is
                    // so the user notices, but don't crash.
                    file.flush().await?;
                    return Ok(());
                }
                file.flush().await?;
                // Rewrite header with real sizes.
                let h = wav_header(sample_rate, bytes_written);
                use tokio::io::AsyncSeekExt;
                file.seek(std::io::SeekFrom::Start(0))
                    .await
                    .with_context(|| format!("seeking {}", path.display()))?;
                file.write_all(&h).await?;
                file.flush().await?;
            }
        }
        Ok(())
    }
}
