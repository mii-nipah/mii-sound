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
pub const OP_TTS_STREAM: u8 = 3;
/// Worker-only op: a batch of independent TTS items dispatched as a single
/// VoxCPM batched forward pass. Wire format after the standard
/// `version | op` header:
///
///     u32 count
///     repeated `count` times: u32 json_len | json | u32 audio_len | audio
///
/// Response wire format (worker → frontend):
///
///     u32 count
///     repeated `count` times: u8 status | u32 payload_len | payload
///
/// Item order is preserved between request and response.
pub const OP_TTS_BATCH: u8 = 4;

// Status bytes reuse exit codes for their natural meaning.
pub const ST_OK: u8 = 0;
pub const ST_MODEL_MISSING: u8 = 2;
pub const ST_GENERATION_FAILED: u8 = 3;
pub const ST_BAD_REQUEST: u8 = 4;
#[allow(dead_code)] // protocol completeness; reserved for future use
pub const ST_UNKNOWN: u8 = 5;

// Streaming response frame kinds. Only used after the client sends OP_TTS_STREAM;
// the wire is then a sequence of frames terminated by KIND_END or KIND_ERROR.
//
// Frame layout: u8 kind | u32 payload_len LE | payload
//   KIND_HEADER payload: u32 sample_rate LE
//   KIND_CHUNK  payload: raw f32 LE samples (mono, model sample rate)
//   KIND_END    payload: empty
//   KIND_ERROR  payload: u8 status (one of ST_*) followed by utf-8 message bytes
pub const KIND_HEADER: u8 = 0;
pub const KIND_CHUNK: u8 = 1;
pub const KIND_END: u8 = 2;
pub const KIND_ERROR: u8 = 3;

// Cap incoming payloads to something generous but bounded.
const MAX_JSON: u32 = 1024 * 1024; // 1 MiB JSON
const MAX_AUDIO: u32 = 256 * 1024 * 1024; // 256 MiB audio
const MAX_PAYLOAD: u32 = 512 * 1024 * 1024; // 512 MiB response
const MAX_TOKEN: u32 = 4096;
/// Hard upper bound on items in one batch envelope. Generous; the configured
/// `--parallel` ceiling is normally well below this.
const MAX_BATCH: u32 = 1024;

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
    let op = read_op(r).await?;
    read_request_body(r, op).await
}

/// Read just the `version | op` header. Used by the worker so it can
/// dispatch on op (single vs. batch) before reading the body.
pub async fn read_op<R: AsyncRead + Unpin>(r: &mut R) -> Result<u8> {
    let version = r.read_u8().await?;
    if version != PROTO_VERSION {
        bail!("unsupported protocol version {version}");
    }
    r.read_u8().await.map_err(|e| anyhow!("reading op: {e}"))
}

/// Read the body of a single (non-batch) request, given the op already
/// pulled from the wire.
pub async fn read_request_body<R: AsyncRead + Unpin>(r: &mut R, op: u8) -> Result<Request> {
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

/// One element of a batch request. `json` is a serialized [`TtsRequest`];
/// `audio` is the optional inline reference clip, identical to the
/// single-request `Request::audio` field.
#[derive(Debug, Clone)]
pub struct BatchItem {
    pub json: Bytes,
    pub audio: Option<Bytes>,
}

/// Write a batch envelope to the worker pipe. Caller is responsible for
/// having written nothing else to `w` since the previous full
/// request/response.
pub async fn write_batch_request<W: AsyncWrite + Unpin>(
    w: &mut W,
    items: &[BatchItem],
) -> Result<()> {
    if items.len() as u64 > MAX_BATCH as u64 {
        bail!("batch too large: {}", items.len());
    }
    w.write_u8(PROTO_VERSION).await?;
    w.write_u8(OP_TTS_BATCH).await?;
    w.write_u32_le(items.len() as u32).await?;
    for item in items {
        w.write_u32_le(item.json.len() as u32).await?;
        w.write_all(&item.json).await?;
        let audio = item.audio.as_ref().map(|b| b.as_ref()).unwrap_or(&[]);
        w.write_u32_le(audio.len() as u32).await?;
        if !audio.is_empty() {
            w.write_all(audio).await?;
        }
    }
    w.flush().await?;
    Ok(())
}

/// Read the body of a batch request (after [`read_op`] returned
/// [`OP_TTS_BATCH`]).
pub async fn read_batch_request_body<R: AsyncRead + Unpin>(r: &mut R) -> Result<Vec<BatchItem>> {
    let count = r.read_u32_le().await?;
    if count > MAX_BATCH {
        bail!("batch frame too large: {count}");
    }
    let mut items = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let json_len = r.read_u32_le().await?;
        if json_len > MAX_JSON {
            bail!("batch json frame too large: {json_len}");
        }
        let mut json = vec![0u8; json_len as usize];
        r.read_exact(&mut json).await?;
        let audio_len = r.read_u32_le().await?;
        if audio_len > MAX_AUDIO {
            bail!("batch audio frame too large: {audio_len}");
        }
        let audio = if audio_len == 0 {
            None
        } else {
            let mut buf = vec![0u8; audio_len as usize];
            r.read_exact(&mut buf).await?;
            Some(Bytes::from(buf))
        };
        items.push(BatchItem {
            json: Bytes::from(json),
            audio,
        });
    }
    Ok(items)
}

/// Write a batch response to the frontend (worker → frontend).
pub async fn write_batch_response<W: AsyncWrite + Unpin>(
    w: &mut W,
    responses: &[Response],
) -> Result<()> {
    w.write_u32_le(responses.len() as u32).await?;
    for resp in responses {
        w.write_u8(resp.status).await?;
        w.write_u32_le(resp.payload.len() as u32).await?;
        if !resp.payload.is_empty() {
            w.write_all(&resp.payload).await?;
        }
    }
    w.flush().await?;
    Ok(())
}

/// Read a batch response from the worker pipe.
pub async fn read_batch_response<R: AsyncRead + Unpin>(r: &mut R) -> Result<Vec<Response>> {
    let count = r.read_u32_le().await?;
    if count > MAX_BATCH {
        bail!("batch response too large: {count}");
    }
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let status = r.read_u8().await?;
        let len = r.read_u32_le().await?;
        if len > MAX_PAYLOAD {
            bail!("batch response payload too large: {len}");
        }
        let mut buf = vec![0u8; len as usize];
        if len > 0 {
            r.read_exact(&mut buf).await?;
        }
        out.push(Response {
            status,
            payload: Bytes::from(buf),
        });
    }
    Ok(out)
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

#[derive(Debug)]
pub struct StreamFrame {
    pub kind: u8,
    pub payload: Bytes,
}

pub async fn write_stream_frame<W: AsyncWrite + Unpin>(
    w: &mut W,
    frame: &StreamFrame,
) -> Result<()> {
    w.write_u8(frame.kind).await?;
    w.write_u32_le(frame.payload.len() as u32).await?;
    if !frame.payload.is_empty() {
        w.write_all(&frame.payload).await?;
    }
    w.flush().await?;
    Ok(())
}

pub async fn read_stream_frame<R: AsyncRead + Unpin>(r: &mut R) -> Result<StreamFrame> {
    let kind = r.read_u8().await?;
    let len = r.read_u32_le().await?;
    if len > MAX_PAYLOAD {
        bail!("stream frame too large: {len}");
    }
    let mut buf = vec![0u8; len as usize];
    if len > 0 {
        r.read_exact(&mut buf).await?;
    }
    Ok(StreamFrame {
        kind,
        payload: Bytes::from(buf),
    })
}

/// Build a KIND_HEADER frame payload from a sample rate.
pub fn header_payload(sample_rate: u32) -> Bytes {
    Bytes::copy_from_slice(&sample_rate.to_le_bytes())
}

/// Decode a KIND_HEADER frame payload back into its sample rate.
pub fn parse_header_payload(payload: &[u8]) -> Result<u32> {
    if payload.len() != 4 {
        bail!("invalid header frame payload length: {}", payload.len());
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(payload);
    Ok(u32::from_le_bytes(buf))
}

/// Build a KIND_ERROR frame payload from (status, message).
pub fn error_payload(status: u8, msg: &str) -> Bytes {
    let mut v = Vec::with_capacity(1 + msg.len());
    v.push(status);
    v.extend_from_slice(msg.as_bytes());
    Bytes::from(v)
}

/// Decode a KIND_ERROR frame payload back into (status, message).
pub fn parse_error_payload(payload: &[u8]) -> (u8, String) {
    if payload.is_empty() {
        return (ST_UNKNOWN, String::new());
    }
    let status = payload[0];
    let msg = String::from_utf8_lossy(&payload[1..]).into_owned();
    (status, msg)
}
