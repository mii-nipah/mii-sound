//! Client side: connect to a local socket (interprocess, cross-platform) or
//! TCP, send a request, handle the response.

pub mod tts;

use crate::cli::Cli;
use crate::exit;
use crate::proto::{self, OP_STATUS, Request, Response};
use crate::transport;
use anyhow::{Context, Result};
use bytes::Bytes;
use interprocess::local_socket::tokio::Stream as IpcStream;
use interprocess::local_socket::traits::tokio::Stream as IpcStreamTrait;
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
