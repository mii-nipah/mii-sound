# mii-sound

An utility designed for easy sound generation, fuelled by `voxcpm-rs`.

It's composed of different modules that do different kinds of sounds.

It's made to be very composable and unix-like, run the process, data goes in, data goes out, you chose how to interact with it.

## serve & client
* `mii-sound serve ...` // you run this on your own, it's the setup required for the client invocations later, with `serve` the program enters the server mode (UDS) and will attend to requests
this is good to allow for model reuse, reducing cold start problems for multiple generations
it should also be very lightweight, it only consumes resources when they are actually being used
`mii-sound serve ... --holds 10m` (default) --holds configures the time each resource will be kept loaded after usage, if in this window another call happens it's reset, and if the specified time passes the resource will be automatically unloaded. Format is simple, '10m', '30s', '1h', '3d', etc...

if you don't specify `serve` it's assumed that the process is a client, it will simply do a request to the server, finish it's processing and then die like a normal client utility process

if you want, you can also serve it using the network stack by:
* `mii-sound serve --network=<port>`, pass $TOKEN env var for additional protection
and you can pass `--url=<url:port>` to clients universally for them to be able to connect

when serving, you *need* to pass the model weights path to the command,
```sh
mii-sound serve ... \
  --tts-dir <path> \
  ...
```

optionally, you can also:
* `mii-sound serve --cpu`
to force the models to run using your CPU instead of wgpu

clients can use:
* `mii-sound --status` to check if the server is up and running (0 yes / 1 no)
useful for any kind of frontend that wants to consume the tool
it will also write "running" or "unreachable" to stdout

## inputs
inputs are passed as JSONs (primarily) via stdin on clients, you may also use the `--json` flag to be able to pass them explicitly instead of in stdin

## tts
* `mii-sound tts` is how you access the tts module of mii-sound
by default it expects a stdin with the json that describes the request, the shape is:
```json
{
  "text": "string"
}
```
and it will stdout the resulting audio data

you can specify
* `mii-sound tts --out <path>` in order to output it to a file directly instead

voice cloning is a feature of some kinds of TTS models, you can use by
* `mii-sound tts --voice-clone`
and passing:
```json
{
  "text": "string",
  "reference": "path",
  "continuation": "string" // optional, used for engines that have the possibility ofenhancing the reference with it's text contents
}
```
you can also:
```json
{
  "text": "string",
  "reference": "<>"
}
```
'<>' is a special string that will tell to the client to expect the reference audio file to come after the JSON from stdin

you can configure adherence (CFG) with:
* `mii-sound tts ... --adherence|--cfg|-a <number (default: 2.0)>`
the higher the values more adherence to the prompt, but will increase the chance of artifacts

and you can configure the steps with:
* `mii-sound tts ... --steps|-s <number (default: 10)>`
the higher the value more steps, after a point has diminishing returns (and can be detrimental) and significantly increases generation times

## music
(WIP)

## streaming
the user can:
* `mii-sound tts ... --stream`
to receive the resulting audio as it's generated

## cancellation
the server will consider the generation tied to the socket connection, if the connection dies the generation is cancelled automatically

## formats
input/output formats are model dependent, but the baseline is .wav

## exit codes
0 means success
1 means server unreachable
2 means model/vae not found
3 generation failed / cancelled
4 bad request / validation
5 means unknown error, read stderr for more informations
