//! Relay mode: forward all local-protocol requests to a remote `mii-http`
//! hosted mii-sound server, using a typed client generated at compile time
//! from `mii-sound.http`.
//!
//! The HTTP token comes from `$MII_SOUND_TOKEN` so individual clients on this
//! machine don't have to know it.

use crate::proto::{
    self, KIND_CHUNK, KIND_END, KIND_ERROR, KIND_HEADER, ST_BAD_REQUEST, ST_GENERATION_FAILED,
    ST_MODEL_MISSING, StreamFrame, TtsRequest as WireTtsRequest,
};
use anyhow::{Result, anyhow};
use bytes::Bytes;
use mii_http_client::{ByteStream, Error as HttpError, FilePart};
use tokio::sync::mpsc;

mii_http_client::client! {
    pub struct MiiSoundRelay;
    spec = "mii-sound.http";

    GET /status              as r_status        => String;
    POST /tts                as r_tts           => mii_http_client::Bytes;
    POST /tts/stream         as r_tts_stream    => mii_http_client::ByteStream;
    POST /tts/clone          as r_tts_clone     => mii_http_client::Bytes;
    POST /tts/clone/stream   as r_tts_clone_stream => mii_http_client::ByteStream;
}

pub struct Relay {
    api: MiiSoundRelay,
    /// Logged at startup so we can format clean errors later.
    upstream: String,
}

impl Relay {
    pub fn new(url: &str) -> Result<Self> {
        let upstream = normalize_url(url);
        let mut api = MiiSoundRelay::new(&upstream)
            .map_err(|e| anyhow!("invalid relay url {upstream}: {e}"))?;
        if let Ok(token) = std::env::var("MII_SOUND_TOKEN")
            && !token.is_empty()
        {
            api = api.bearer_token(token);
        } else {
            log::warn!(
                "$MII_SOUND_TOKEN not set; relay will send requests without an Authorization header"
            );
        }
        Ok(Self { api, upstream })
    }

    pub fn upstream(&self) -> &str {
        &self.upstream
    }

    /// Probe the remote /status endpoint. Returns true iff the upstream
    /// reports "running".
    pub async fn status(&self) -> bool {
        match self.api.r_status().await {
            Ok(body) => body.trim() == "running",
            Err(e) => {
                log::warn!("relay status probe failed: {}", http_err(&e));
                false
            }
        }
    }

    /// Forward a non-streaming TTS request and return the audio bytes.
    pub async fn tts(&self, req: &WireTtsRequest, audio: Option<Bytes>) -> RelayResult<Bytes> {
        if needs_clone(req) {
            self.tts_clone(req, audio).await
        } else {
            self.tts_plain(req).await
        }
    }

    async fn tts_plain(&self, req: &WireTtsRequest) -> RelayResult<Bytes> {
        let body = build_plain_body(req);
        let request = RTtsRequest {
            cfg: Some(format_cfg(req.cfg)),
            steps: Some(req.steps as i64),
            body,
        };
        match self.api.r_tts(request).await {
            Ok(bytes) => Ok(Bytes::copy_from_slice(&bytes)),
            Err(e) => Err(map_http_err(e)),
        }
    }

    async fn tts_clone(&self, req: &WireTtsRequest, audio: Option<Bytes>) -> RelayResult<Bytes> {
        let reference = build_reference_part(req, audio)?;
        warn_on_dropped_continuation(req);
        let body = RTtsCloneBody {
            text: req.text.clone(),
            reference,
        };
        let request = RTtsCloneRequest {
            cfg: Some(format_cfg(req.cfg)),
            steps: Some(req.steps as i64),
            body,
        };
        match self.api.r_tts_clone(request).await {
            Ok(bytes) => Ok(Bytes::copy_from_slice(&bytes)),
            Err(e) => Err(map_http_err(e)),
        }
    }

    /// Forward a streaming TTS request, translating the upstream chunked WAV
    /// stream into local-protocol frames (`KIND_HEADER`, `KIND_CHUNK`,
    /// terminal `KIND_END` / `KIND_ERROR`).
    pub async fn tts_stream(
        &self,
        req: &WireTtsRequest,
        audio: Option<Bytes>,
        out: mpsc::Sender<StreamFrame>,
    ) {
        let stream_result = if needs_clone(req) {
            let reference = match build_reference_part(req, audio) {
                Ok(p) => p,
                Err(e) => {
                    let _ = out.send(error_frame(e.status, &e.message)).await;
                    return;
                }
            };
            warn_on_dropped_continuation(req);
            let body = RTtsCloneStreamBody {
                text: req.text.clone(),
                reference,
            };
            self.api
                .r_tts_clone_stream(RTtsCloneStreamRequest {
                    cfg: Some(format_cfg(req.cfg)),
                    steps: Some(req.steps as i64),
                    body,
                })
                .await
        } else {
            let body = build_plain_body(req);
            self.api
                .r_tts_stream(RTtsStreamRequest {
                    cfg: Some(format_cfg(req.cfg)),
                    steps: Some(req.steps as i64),
                    body,
                })
                .await
        };

        let stream = match stream_result {
            Ok(s) => s,
            Err(e) => {
                let mapped = map_http_err(e);
                let _ = out.send(error_frame(mapped.status, &mapped.message)).await;
                return;
            }
        };

        if let Err((status, msg)) = relay_wav_stream(stream, &out).await {
            let _ = out.send(error_frame(status, &msg)).await;
        }
    }
}

// --- helpers ---------------------------------------------------------------

fn needs_clone(req: &WireTtsRequest) -> bool {
    req.inline_reference || req.reference.is_some()
}

fn format_cfg(cfg: f32) -> String {
    // Strip trailing zeros so `2.0` doesn't become `2.0000` (the spec regex
    // accepts both, but tidy is nicer in logs / on the wire).
    let s = format!("{cfg:.4}");
    let trimmed = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if trimmed.is_empty() || trimmed == "-" {
        "0".into()
    } else {
        trimmed
    }
}

fn build_plain_body(req: &WireTtsRequest) -> mii_http_client::serde_json::Value {
    use mii_http_client::serde_json::{Map, Value};
    let mut obj = Map::new();
    obj.insert("text".into(), Value::String(req.text.clone()));
    if let Some(c) = req.continuation.as_ref() {
        obj.insert("continuation".into(), Value::String(c.clone()));
    }
    Value::Object(obj)
}

fn build_reference_part(req: &WireTtsRequest, audio: Option<Bytes>) -> RelayResult<FilePart> {
    if req.inline_reference {
        let bytes = audio.ok_or_else(|| RelayError {
            status: ST_BAD_REQUEST,
            message: "inline reference flagged but no audio bytes received".into(),
        })?;
        return Ok(FilePart::bytes(bytes.to_vec())
            .with_file_name("reference.wav")
            .with_mime("audio/wav"));
    }
    let path = req.reference.as_deref().ok_or_else(|| RelayError {
        status: ST_BAD_REQUEST,
        message: "voice cloning request missing `reference`".into(),
    })?;
    Ok(FilePart::path(path))
}

fn warn_on_dropped_continuation(req: &WireTtsRequest) {
    if req.continuation.is_some() {
        log::warn!(
            "relay: `continuation` is not currently exposed by the HTTP voice-clone endpoint; \
             dropping it for this request"
        );
    }
}

fn normalize_url(input: &str) -> String {
    let s = input.trim();
    if s.starts_with("http://") || s.starts_with("https://") {
        s.to_string()
    } else {
        format!("http://{s}")
    }
}

fn http_err(e: &HttpError) -> String {
    match e {
        HttpError::UnexpectedStatus { status, body } => {
            format!("upstream returned {status}: {body}")
        }
        other => other.to_string(),
    }
}

fn map_http_err(e: HttpError) -> RelayError {
    let message = http_err(&e);
    let status = match &e {
        HttpError::UnexpectedStatus { status, .. } if status.as_u16() == 401 => ST_BAD_REQUEST,
        HttpError::UnexpectedStatus { status, .. } if status.as_u16() == 400 => ST_BAD_REQUEST,
        HttpError::UnexpectedStatus { status, .. } if status.as_u16() == 503 => ST_MODEL_MISSING,
        _ => ST_GENERATION_FAILED,
    };
    RelayError { status, message }
}

fn error_frame(status: u8, msg: &str) -> StreamFrame {
    StreamFrame {
        kind: KIND_ERROR,
        payload: proto::error_payload(status, msg),
    }
}

pub struct RelayError {
    pub status: u8,
    pub message: String,
}
pub type RelayResult<T> = std::result::Result<T, RelayError>;

// --- WAV-stream → frame translation ---------------------------------------
//
// The upstream `/tts/stream` endpoint emits a 16-bit PCM mono WAV with
// streaming-sentinel sizes (`0xFFFFFFFF`). The local protocol expects a
// `KIND_HEADER` (sample rate) followed by `KIND_CHUNK` frames carrying raw
// `f32` LE mono samples, terminated by `KIND_END` / `KIND_ERROR`.
//
// We buffer the first 44 header bytes, parse the sample rate out of the `fmt`
// subchunk, and forward subsequent bytes as freshly-typed sample frames. We
// keep an internal carry buffer in case a chunk boundary lands in the middle
// of an `i16`.

const WAV_HEADER_LEN: usize = 44;
const WAV_SAMPLE_RATE_OFFSET: usize = 24;

async fn relay_wav_stream(
    mut stream: ByteStream,
    out: &mpsc::Sender<StreamFrame>,
) -> std::result::Result<(), (u8, String)> {
    let mut header_buf: Vec<u8> = Vec::with_capacity(WAV_HEADER_LEN);
    let mut header_done = false;
    let mut carry: u8 = 0;
    let mut have_carry = false;

    loop {
        let chunk = match stream.chunk().await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            Err(e) => {
                return Err((ST_GENERATION_FAILED, format!("upstream stream: {e}")));
            }
        };
        if chunk.is_empty() {
            continue;
        }
        let mut pos = 0;
        if !header_done {
            let need = WAV_HEADER_LEN - header_buf.len();
            let take = need.min(chunk.len());
            header_buf.extend_from_slice(&chunk[..take]);
            pos = take;
            if header_buf.len() == WAV_HEADER_LEN {
                let sample_rate =
                    parse_wav_sample_rate(&header_buf).map_err(|e| (ST_GENERATION_FAILED, e))?;
                let header_frame = StreamFrame {
                    kind: KIND_HEADER,
                    payload: proto::header_payload(sample_rate),
                };
                if out.send(header_frame).await.is_err() {
                    // Receiver gone; drop quietly.
                    return Ok(());
                }
                header_done = true;
            }
        }
        if header_done && pos < chunk.len() {
            let body = &chunk[pos..];
            forward_pcm_chunk(body, &mut carry, &mut have_carry, out).await;
        }
    }

    if !header_done {
        return Err((
            ST_GENERATION_FAILED,
            format!(
                "upstream stream ended before WAV header arrived ({} of {WAV_HEADER_LEN} bytes received)",
                header_buf.len()
            ),
        ));
    }

    if have_carry {
        log::warn!("relay: dropping trailing odd byte from upstream PCM stream");
    }

    let _ = out
        .send(StreamFrame {
            kind: KIND_END,
            payload: Bytes::new(),
        })
        .await;
    Ok(())
}

async fn forward_pcm_chunk(
    bytes: &[u8],
    carry: &mut u8,
    have_carry: &mut bool,
    out: &mpsc::Sender<StreamFrame>,
) {
    let mut idx = 0usize;
    let mut samples: Vec<f32> = Vec::with_capacity(bytes.len() / 2 + 1);
    if *have_carry && !bytes.is_empty() {
        let s = i16::from_le_bytes([*carry, bytes[0]]);
        samples.push(s as f32 / 32768.0);
        idx = 1;
        *have_carry = false;
    }
    let pair_end = idx + ((bytes.len() - idx) & !1);
    let mut i = idx;
    while i < pair_end {
        let s = i16::from_le_bytes([bytes[i], bytes[i + 1]]);
        samples.push(s as f32 / 32768.0);
        i += 2;
    }
    if i < bytes.len() {
        *carry = bytes[i];
        *have_carry = true;
    }
    if samples.is_empty() {
        return;
    }
    let mut payload = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        payload.extend_from_slice(&s.to_le_bytes());
    }
    let _ = out
        .send(StreamFrame {
            kind: KIND_CHUNK,
            payload: Bytes::from(payload),
        })
        .await;
}

fn parse_wav_sample_rate(header: &[u8]) -> std::result::Result<u32, String> {
    if header.len() < WAV_SAMPLE_RATE_OFFSET + 4 {
        return Err("upstream WAV header too short".into());
    }
    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err("upstream did not return a RIFF/WAVE stream".into());
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&header[WAV_SAMPLE_RATE_OFFSET..WAV_SAMPLE_RATE_OFFSET + 4]);
    Ok(u32::from_le_bytes(buf))
}

// `Result<()>` from anyhow is exported above so callers in this module can
// keep using it; the `?`-propagation across the relay public surface returns
// `RelayResult<T>` instead.
#[allow(dead_code)]
fn _result_marker(_: Result<()>) {}
