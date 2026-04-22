mod cli;
mod client;
mod exit;
mod proto;
mod server;
mod transport;

use clap::Parser;
use cli::{Cli, Command};

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let parsed = Cli::parse();

    if parsed.status {
        let code = client::run_status(&parsed).await;
        std::process::exit(code);
    }

    match parsed.command.clone() {
        Some(Command::Serve(args)) => {
            let socket = parsed.socket.clone();
            match server::run(args, socket).await {
                Ok(()) => std::process::exit(exit::SUCCESS),
                Err(e) => {
                    eprintln!("mii-sound serve: {e:#}");
                    std::process::exit(exit::UNKNOWN);
                }
            }
        }
        Some(Command::Tts(args)) => client::tts::run(&parsed, &args).await,
        None => {
            eprintln!(
                "mii-sound: no subcommand given (use `tts` or `serve`, or pass --status)"
            );
            std::process::exit(exit::BAD_REQUEST);
        }
    }
}
