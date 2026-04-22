//! Wire protocol shared between client and server.
//!
//! Frame layout (after optional TCP token frame):
//! Request:  u8 version | u8 op | u32 json_len (LE) | json | u32 audio_len (LE) | audio
//! Response: u8 status  | u32 payload_len (LE) | payload
//!
//! For TCP transport an extra leading `u32 token_len | token_bytes` frame is
//! sent by the client and validated by the server.

use anyhow::{Result, anyhow, bail};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const PROTO_VERSION: u8 = 1;

pub const OP_TTS: u8 = 1;
pub const OP_STATUS: u8 = 2;

// Status bytes reuse exit codes for their natural meaning.
pub const ST_OK: u8 = 0;
pub const ST_MODEL_MISSING: u8 = 2;
pub const ST_GENERATION_FAILED: u8 = 3;
pub const ST_BAD_REQUEST: u8 = 4;
#[allow(dead_code)] // protocol completeness; reserved for future use
pub const ST_UNKNOWN: u8 = 5;

// Cap incoming payloads to something generous but bounded.
const MAX_JSON: u32 = 1024 * 1024; // 1 MiB JSON
const MAX_AUDIO: u32 = 256 * 1024 * 1024; // 256 MiB audio
const MAX_PAYLOAD: u32 = 512 * 1024 * 1024; // 512 MiB response
const MAX_TOKEN: u32 = 4096;

/// Wire-level TTS request. The user-facing JSON only carries `text`,
/// `reference`, and `continuation` (per specs.md); the client merges its CLI
/// flags (`cfg`, `steps`, `voice_clone`) into this struct before sending.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRequest {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation: Option<String>,
    #[serde(default = "default_cfg")]
    pub cfg: f32,
    #[serde(default = "default_steps")]
    pub steps: u32,
    /// True iff the inline audio frame holds the reference clip
    /// (i.e. user wrote `"reference": "<>"`).
    #[serde(default)]
    pub inline_reference: bool,
}

fn default_cfg() -> f32 {
    2.0
}
fn default_steps() -> u32 {
    10
}

#[derive(Debug)]
pub struct Request {
    pub op: u8,
    pub json: Bytes,
    pub audio: Option<Bytes>,
}

#[derive(Debug)]
pub struct Response {
    pub status: u8,
    pub payload: Bytes,
}

pub async fn write_token<W: AsyncWrite + Unpin>(w: &mut W, token: &str) -> Result<()> {
    let bytes = token.as_bytes();
    if bytes.len() as u64 > MAX_TOKEN as u64 {
        bail!("token too long");
    }
    w.write_u32_le(bytes.len() as u32).await?;
    w.write_all(bytes).await?;
    Ok(())
}

pub async fn read_token<R: AsyncRead + Unpin>(r: &mut R) -> Result<String> {
    let len = r.read_u32_le().await?;
    if len > MAX_TOKEN {
        bail!("token frame too large");
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).await?;
    String::from_utf8(buf).map_err(|e| anyhow!("token not utf-8: {e}"))
}

pub async fn write_request<W: AsyncWrite + Unpin>(w: &mut W, req: &Request) -> Result<()> {
    w.write_u8(PROTO_VERSION).await?;
    w.write_u8(req.op).await?;
    w.write_u32_le(req.json.len() as u32).await?;
    w.write_all(&req.json).await?;
    let audio = req.audio.as_ref().map(|b| b.as_ref()).unwrap_or(&[]);
    w.write_u32_le(audio.len() as u32).await?;
    if !audio.is_empty() {
        w.write_all(audio).await?;
    }
    w.flush().await?;
    Ok(())
}

pub async fn read_request<R: AsyncRead + Unpin>(r: &mut R) -> Result<Request> {
    let version = r.read_u8().await?;
    if version != PROTO_VERSION {
        bail!("unsupported protocol version {version}");
    }
    let op = r.read_u8().await?;
    let json_len = r.read_u32_le().await?;
    if json_len > MAX_JSON {
        bail!("json frame too large: {json_len}");
    }
    let mut json = vec![0u8; json_len as usize];
    r.read_exact(&mut json).await?;
    let audio_len = r.read_u32_le().await?;
    if audio_len > MAX_AUDIO {
        bail!("audio frame too large: {audio_len}");
    }
    let audio = if audio_len == 0 {
        None
    } else {
        let mut buf = vec![0u8; audio_len as usize];
        r.read_exact(&mut buf).await?;
        Some(Bytes::from(buf))
    };
    Ok(Request {
        op,
        json: Bytes::from(json),
        audio,
    })
}

pub async fn write_response<W: AsyncWrite + Unpin>(w: &mut W, resp: &Response) -> Result<()> {
    w.write_u8(resp.status).await?;
    w.write_u32_le(resp.payload.len() as u32).await?;
    if !resp.payload.is_empty() {
        w.write_all(&resp.payload).await?;
    }
    w.flush().await?;
    Ok(())
}

pub async fn read_response<R: AsyncRead + Unpin>(r: &mut R) -> Result<Response> {
    let status = r.read_u8().await?;
    let len = r.read_u32_le().await?;
    if len > MAX_PAYLOAD {
        bail!("response payload too large: {len}");
    }
    let mut buf = vec![0u8; len as usize];
    if len > 0 {
        r.read_exact(&mut buf).await?;
    }
    Ok(Response {
        status,
        payload: Bytes::from(buf),
    })
}
