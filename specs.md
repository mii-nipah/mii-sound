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
  --tts-model <path.safetensors> \
  --tts-vae <path.pth/safetensors> \
  ...
```

clients can use:
* `mii-sound --status` to check if the server is up and running (0 yes / 1 no)
useful for any kind of frontend that wants to consume the tool

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

## music
(WIP)

## streaming
* currently unimplemented, under active consideration

## cancellation
the server will consider the generation tied to the socket connection, if the connection dies the generation is cancelled automatically

## exit codes
0 means success
1 means the server was not found (clients) / the model/vae was not found (servers)
2 means unknown error, read stderr for more informations
