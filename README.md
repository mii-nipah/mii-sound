# mii-sound

> A small, composable, unix-y sound generation utility — fueled by [`voxcpm-rs`](https://crates.io/crates/voxcpm-rs).

`mii-sound` is a CLI you point a JSON request at and get audio back. It is designed to slot
into shell pipelines, scripts, and frontends without ceremony: data goes in, audio goes out,
and you choose how to wire it up. A long-lived `serve` mode keeps models warm between
invocations so cold starts don't hurt.

```sh
echo '{"text":"hello, world"}' | mii-sound tts --out hello.wav
```

---

## Index

- [Features](#features)
- [Quick start](#quick-start)
- [Usage](#usage)
  - [Server](#server)
  - [TTS client](#tts-client)
  - [Voice cloning](#voice-cloning)
  - [Streaming](#streaming)
  - [Status checks](#status-checks)
- [Architecture](#architecture)
- [Exit codes](#exit-codes)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Related projects](#related-projects)

---

## Features

- **Composable.** Stdin in, audio out. Pipe it anywhere.
- **Warm models.** Run `mii-sound serve` once; clients reuse loaded weights.
- **Auto-unload.** Idle resources are dropped after a configurable hold window.
- **Local or networked.** Talks over a Unix socket by default, or TCP with a token.
- **Backends.** Runs on `wgpu` by default, with a `--cpu` fallback.
- **Voice cloning.** Bring your own reference audio for supported engines.
- **Streaming.** `--stream` to start hearing audio before the utterance is done.
- **Cancellation.** Drop the connection, cancel the generation. No knobs needed.

## Quick start

Install with cargo:

```sh
cargo install mii-sound
```

Start the server (in one terminal) pointing at a VoxCPM2 model directory:

```sh
mii-sound serve --tts-dir /path/to/voxcpm2
```

Then, from anywhere:

```sh
echo '{"text":"this is mii-sound speaking"}' \
  | mii-sound tts --out greeting.wav
```

That's it. Play `greeting.wav` with any audio player.

## Usage

### Server

```sh
mii-sound serve --tts-dir <path> [--cpu] [--holds 10m] [--network <port>] [--quiet]
```

- `--tts-dir <path>` — VoxCPM2 model directory (`config.json`, `tokenizer.json`,
  `model.safetensors`, `audiovae.pth`).
- `--cpu` — force CPU backend instead of `wgpu`.
- `--holds <duration>` — keep idle resources loaded for this long. Examples:
  `30s`, `10m` (default), `1h`, `3d`. Each use resets the timer.
- `--network <port>` — listen on TCP. Set the `TOKEN` env var to require auth.
- `--quiet` — suppress per-request and lifecycle logs.

Clients connect to the local UDS by default. Override with `--socket <path>`
or, for TCP servers, `--url host:port`.

### TTS client

```sh
mii-sound tts [--out <path>] [--json <inline>] [--cfg <f>] [--steps <n>] [--stream]
```

Request shape (stdin or `--json`):

```json
{ "text": "hello, world" }
```

- `--out <path>` — write to a file instead of stdout.
- `--json <inline>` — pass the JSON inline rather than via stdin.
- `--cfg / --adherence / -a <number>` — classifier-free guidance (default `2.0`).
  Higher = more adherence, more artifact risk.
- `--steps / -s <number>` — diffusion steps (default `10`). Diminishing returns
  past a point, and longer generations.
- `--stream` — emit audio as it is generated. See [Streaming](#streaming).

Output format is model-dependent; the baseline is `.wav`.

### Voice cloning

Pass `--voice-clone` and include a `reference` in the request:

```json
{
  "text": "speak in this voice",
  "reference": "/path/to/sample.wav",
  "continuation": "optional transcript of the reference"
}
```

To stream the reference over stdin instead of pointing at a file, use the
sentinel `"<>"`:

```json
{ "text": "...", "reference": "<>" }
```

…and append the raw reference audio bytes after the JSON.

### Streaming

Pass `--stream` to receive audio as it is produced instead of waiting for the
whole utterance:

```sh
echo '{"text":"streaming hello!"}' \
  | mii-sound tts --stream \
  | aplay -q
```

The output is a single 16-bit PCM mono WAV at the model's sample rate. On
stdout the RIFF and `data` chunk sizes are written as `0xFFFFFFFF` — the
conventional "size unknown / streaming" sentinel that `aplay`, `ffplay`,
`mpv`, and most decoders accept. With `--out <path>`, the header is finalized
with the real sizes once the stream completes, so the resulting file is a
standard, fully-valid WAV.

Dropping the client (Ctrl-C, closed pipe) cancels the generation server-side,
just like with non-streaming requests.

### Status checks

```sh
mii-sound --status
# prints "running" or "unreachable"; exits 0 or 1
```

Handy for frontends and health probes.

## Architecture

```
┌──────────────┐    UDS / TCP   ┌─────────────────┐    stdio    ┌──────────────┐
│ mii-sound    │ ─────────────► │ mii-sound serve │ ──────────► │ tts-worker   │
│ tts (client) │ ◄───────────── │  (broker)       │ ◄────────── │ (model host) │
└──────────────┘   audio bytes  └─────────────────┘  protocol   └──────────────┘
```

- **Client** (`src/client/`) — parses the request, opens a connection, streams
  audio back to stdout or `--out`.
- **Broker** (`src/server/`) — accepts client connections, owns the worker
  lifecycle, enforces `--holds` idle unloading, and forwards requests.
- **Worker** (`src/worker.rs`) — a separate long-lived process that hosts the
  model and speaks a small framed protocol over stdio. Spawned automatically;
  not meant to be invoked directly.
- **Transport / proto** (`src/transport.rs`, `src/proto.rs`) — wire format
  shared by all three.
- **Synth** (`src/synth.rs`) — the actual call into `voxcpm-rs`.

Cancellation is implicit: a generation is bound to its connection, so closing
the socket cancels the work.

## Exit codes

| Code | Meaning                                    |
|-----:|--------------------------------------------|
| `0`  | success                                    |
| `1`  | server unreachable                         |
| `2`  | model / vae not found                      |
| `3`  | generation failed or cancelled             |
| `4`  | bad request / validation error             |
| `5`  | unknown error (check stderr)               |

## Roadmap

- [ ] **Music** module
- [ ] More TTS backends

See [specs.md](specs.md) for the living design notes.

## Contributing

Issues and PRs are welcome. A few light guidelines:

- Keep changes small and focused; one concern per PR.
- Run `cargo fmt` and `cargo clippy` before pushing.
- For non-trivial features, open an issue first so we can chat about the shape.
- Be kind. Assume good intent.

## Related projects

- [`voxcpm-rs`](https://crates.io/crates/voxcpm-rs) — the TTS engine powering `mii-sound`.
- [`burn`](https://burn.dev) — the ML framework underneath.
