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
  - [RunPod Docker](#runpod-docker)
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
- **Batching.** `serve --parallel <N>` groups concurrent requests into one
  batched forward pass for substantially higher GPU throughput.

## Quick start

Install a prebuilt binary from GitHub Releases. With the GitHub CLI, grab the
latest Linux archive like this:

```sh
gh release download --repo mii-nipah/mii-sound --pattern 'mii-sound-*-linux-gnu.tar.gz'
tar -xzf mii-sound-*.tar.gz
sudo install -Dm755 mii-sound-*/mii-sound /usr/local/bin/mii-sound
```

crates.io cannot carry this repo's Cargo patches, so its install path is the
regular WGPU build without the Vulkan bf16 patch:

```sh
cargo install mii-sound --locked --no-default-features
```

If you prefer building the fast Vulkan path locally with the same patches,
install from git instead:

```sh
cargo install --git https://github.com/mii-nipah/mii-sound --locked
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
mii-sound serve --tts-dir <path> [--cpu] [--holds 10m] [--network <port>] [--parallel <N>] [--batch-window 300ms] [--quiet]
```

- `--tts-dir <path>` — VoxCPM2 model directory (`config.json`, `tokenizer.json`,
  `model.safetensors`, `audiovae.pth`).
- `--cpu` — force CPU backend instead of `wgpu`.
- `--holds <duration>` — keep idle resources loaded for this long. Examples:
  `30s`, `10m` (default), `1h`, `3d`. Each use resets the timer.
- `--network <port>` — listen on TCP. Set the `TOKEN` env var to require auth.
- `--parallel <N>` — process up to `N` concurrent requests in a single
  batched VoxCPM forward pass (default `1`, no batching). The server holds
  a short grace window for additional requests to arrive before
  dispatching; requests beyond `N`, or that arrive after the window closes,
  are queued and grouped into the next batch. Items only batch together
  when their `cfg` and `steps` match. Sweet spot is hardware-bound; on
  modern GPUs `4`–`8` typically yields 2–3× throughput over `1`.
- `--batch-window <duration>` — how long the server waits for additional
  requests to fill up a batch once one is pending. Same duration format as
  `--holds` (e.g. `300ms`, `1s`). Default `300ms`. Only meaningful when
  `--parallel > 1`.
- `--quiet` — suppress per-request and lifecycle logs.

Clients connect to the local UDS by default. Override with `--socket <path>`
or, for TCP servers, `--url host:port`.

### RunPod Docker

The repo includes a RunPod-oriented container that keeps `mii-sound serve`
warm on the local socket and exposes the `mii-sound.http` facade through
`mii-http`:

```sh
docker build -t mii-sound-runpod .
```

For local smoke testing with Docker + NVIDIA runtime:

```sh
docker run --rm --gpus all \
  -p 7000:7000 \
  -e TOKEN=dev-token \
  -v "$PWD/.models:/workspace/models" \
  mii-sound-runpod
```

Then hit the HTTP facade:

```sh
curl -H "Authorization: Bearer dev-token" \
  http://localhost:7000/sound/v1/status

curl -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  --data '{"text":"hello from RunPod"}' \
  http://localhost:7000/sound/v1/tts \
  -o runpod.wav
```

On RunPod:

- Use the built image in a Pod template.
- In the Model field, set `openbmb/VoxCPM2` so RunPod pre-caches the Hugging
  Face model for Serverless workers.
- For Pods, expose HTTP port `7000` and use the HTTP proxy URL.
- For Serverless load balancing endpoints, expose HTTP port `7000` and health
  port `7001`. The container serves `/ping` on the health port and the API on
  `/sound/v1/...`.
- Set `TOKEN` as an environment variable. If omitted, the container generates
  a transient token and prints it once in the logs.
- Attach a network volume at `/workspace` if you want fallback downloads to
  survive Pod restarts.

The startup script prefers RunPod's cached Hugging Face snapshot for
`openbmb/VoxCPM2`, then falls back to baked weights, then to downloading into
`/workspace/models/VoxCPM2`. `MII_SOUND_MODEL_DIR` overrides that lookup when
you want to point at a specific complete model directory. Useful environment
variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TOKEN` | generated | Bearer token required by the HTTP facade. |
| `PORT` | `7000` | Main HTTP port for `mii-http`. |
| `MII_SOUND_HTTP_PORT` | unset | Overrides `PORT` for the HTTP facade. |
| `PORT_HEALTH` | `7001` | RunPod load-balancer health port serving `/ping`. |
| `MII_SOUND_PORT` | unset | Legacy alias used as the HTTP port when `PORT` is unset. |
| `MII_SOUND_MODEL_REPO` | `openbmb/VoxCPM2` | Hugging Face repo to download. |
| `MII_SOUND_MODEL_DIR` | unset | Explicit local model directory; disables automatic RunPod cache selection. |
| `MII_SOUND_USE_RUNPOD_MODEL_CACHE` | `1` | Set to `0` to skip RunPod's cached Hugging Face snapshot. |
| `MII_SOUND_HF_CACHE_ROOT` | `/runpod-volume/huggingface-cache/hub` | Hugging Face cache root used by RunPod model caching. |
| `MII_SOUND_HOLDS` | `2h` | How long to keep the worker/model warm after use. |
| `MII_SOUND_CPU` | unset | Set to `1` to force CPU mode. |
| `MII_SOUND_SKIP_MODEL_DOWNLOAD` | unset | Set to `1` to require a pre-mounted model. |
| `MII_HTTP_QUIET` | `1` | Set to `0` to let `mii-http` print request logs. |
| `HF_TOKEN` | unset | Optional Hugging Face token for downloads. |

After the Pod starts, use the HTTP proxy URL shown by RunPod:

```sh
curl -H "Authorization: Bearer <same token>" \
  https://<pod-id>-7000.proxy.runpod.net/sound/v1/status

curl -H "Authorization: Bearer <same token>" \
  -H "Content-Type: application/json" \
  --data '{"text":"hello from RunPod"}' \
  https://<pod-id>-7000.proxy.runpod.net/sound/v1/tts \
  -o runpod.wav
```

For a Serverless load balancing endpoint, use the endpoint host with the same
paths:

```sh
curl -H "Authorization: Bearer <same token>" \
  https://<endpoint-id>.api.runpod.ai/sound/v1/status
```

If you prefer a larger image that already contains the weights, build with:

```sh
docker build \
  --build-arg DOWNLOAD_MODEL=1 \
  -t mii-sound-runpod:voxcpm2 .
```

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
