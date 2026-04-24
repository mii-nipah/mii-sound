//! TTS client subcommand.

use crate::cli::{Cli, TtsArgs};
use crate::client::{
    Conn, connect, fail_bad_request, fail_unknown, fail_unreachable, handle_response_status,
    read_all_stdin, send_recv, split_json_and_rest, status_to_exit, stream_request,
    write_payload_out,
};
use crate::proto::{OP_TTS, OP_TTS_STREAM, Request, TtsRequest};
use bytes::Bytes;

const INLINE_REFERENCE_SENTINEL: &str = "<>";

pub async fn run(cli: &Cli, args: &TtsArgs) -> ! {
    // 1. Acquire request JSON (and possibly inline audio bytes).
    let (json_value, audio_tail) = if let Some(s) = &args.json {
        match serde_json::from_str::<serde_json::Value>(s) {
            Ok(v) => (v, Vec::new()),
            Err(e) => fail_bad_request(format!("--json is not valid JSON: {e}")),
        }
    } else {
        let raw = match read_all_stdin().await {
            Ok(b) => b,
            Err(e) => fail_bad_request(format!("failed to read stdin: {e}")),
        };
        if raw.is_empty() {
            fail_bad_request("no JSON received on stdin (and no --json given)");
        }
        match split_json_and_rest(&raw) {
            Ok((v, rest)) => (v, rest.to_vec()),
            Err(e) => fail_bad_request(e),
        }
    };

    // 2. Parse into our typed shape (allowing the user-facing fields).
    #[derive(serde::Deserialize)]
    struct UserReq {
        text: String,
        #[serde(default)]
        reference: Option<String>,
        #[serde(default)]
        continuation: Option<String>,
    }
    let user: UserReq = match serde_json::from_value(json_value) {
        Ok(v) => v,
        Err(e) => fail_bad_request(format!("invalid request shape: {e}")),
    };

    if user.text.trim().is_empty() {
        fail_bad_request("`text` is required and must be non-empty");
    }

    // Voice cloning rules.
    if args.voice_clone && user.reference.is_none() {
        fail_bad_request("--voice-clone requires a `reference` field in the request JSON");
    }
    if !args.voice_clone && user.reference.is_some() {
        fail_bad_request("`reference` set but --voice-clone not specified");
    }

    let inline = user.reference.as_deref() == Some(INLINE_REFERENCE_SENTINEL);
    let audio_bytes = if inline {
        if audio_tail.is_empty() {
            fail_bad_request("`reference` is `<>` but no audio bytes followed the JSON on stdin");
        }
        Some(Bytes::from(audio_tail))
    } else {
        // Allow trailing whitespace (e.g. the newline from `echo`).
        if audio_tail.iter().any(|b| !b.is_ascii_whitespace()) {
            fail_bad_request("trailing bytes after JSON but `reference` is not `<>`");
        }
        None
    };

    // 3. Build wire request.
    let wire = TtsRequest {
        text: user.text,
        reference: if inline { None } else { user.reference },
        continuation: user.continuation,
        cfg: args.cfg,
        steps: args.steps,
        inline_reference: inline,
    };
    let json = match serde_json::to_vec(&wire) {
        Ok(b) => b,
        Err(e) => fail_unknown(format!("failed to serialize request: {e}")),
    };

    // 4. Connect + send.
    let mut conn: Conn = match connect(cli).await {
        Ok(c) => c,
        Err(e) => fail_unreachable(e),
    };
    let req = Request {
        op: if args.stream { OP_TTS_STREAM } else { OP_TTS },
        json: Bytes::from(json),
        audio: audio_bytes,
    };

    if args.stream {
        match stream_request(&mut conn, &req, args.out.as_deref()).await {
            Ok(()) => std::process::exit(crate::exit::SUCCESS),
            Err((status, msg)) => {
                if !msg.is_empty() {
                    eprintln!("mii-sound: server error: {msg}");
                }
                std::process::exit(status_to_exit(status));
            }
        }
    }

    let resp = match send_recv(&mut conn, &req).await {
        Ok(r) => r,
        Err(e) => fail_unreachable(e),
    };

    handle_response_status(&resp);

    // 5. Write payload out.
    if let Err(e) = write_payload_out(args.out.as_deref(), &resp.payload).await {
        fail_unknown(e);
    }
    std::process::exit(crate::exit::SUCCESS);
}
