# Fixed listening demo

The listening page offers three new English sentences and four speech systems.
It plays 12 fixed WAV recordings. It does not run inference on the host.

The page uses the HiFi-GAN ResBlock2 baseline, Parallel-IR, Sequential-IR and Piper.
The recordings are demonstration samples, not historical Harvard research audio or a listening test.
No quality score, timing benchmark or quality-based sample selection was performed.

## Provenance and terms

[Sample record](listening-samples.json) identifies every WAV and source ONNX graph by SHA-256.
The graphs came from the public inference repository at revision
`f760e85a45b84c217091acf963837ffc817bad8d`.
Generation used the existing Linux amd64 image under local Docker emulation.
The exact software versions and processing are recorded with the samples.

Generated audio uses CC BY 4.0 only for rights held by the project team.
See [audio terms](../licenses/Demo-generated-audio-CC-BY-4.0.md).
The author previously selected this scoped license for generated audio; it is reused for these new demonstration samples.
Source code, model weights and external datasets retain their separate terms.
Harvard sentences and historical WAVs are excluded from this deployment.
Original training checkpoints and logs remain private.

## Hosting

`demo/listen/` contains the static frontend shared by both hosts.
Hugging Face Static Spaces hosts the frontend and audio.
GitHub Pages hosts the frontend and reads pinned Hugging Face audio URLs.
Audio files and weights stay outside Git.

The publication receipt records the actual URLs, revisions and deployment checks.
The GitHub Pages workflow publishes only `demo/listen/`.
Neither host needs a model inference service or a paid compute instance.

## Reproduce generation

`scripts/demo/generate_listening_samples.py` records the original generation method.
Run it inside the prepared inference image with identified models mounted read-only.
Mount an empty output directory at `/generated` and set `HIFIMOBINET_MODEL_DIR` to the model directory.
The script refuses to overwrite WAV files.
Synthesis is stochastic. Regeneration can produce different bytes; do not replace the published identity record with newly generated files.

## Local preview

Serve `demo/listen/` with a local HTTP server after obtaining its published catalog.
Do not open `index.html` through a file URL because the browser must fetch `catalog.json`.

```sh
python -m http.server 8873 --bind 127.0.0.1 --directory demo/listen
```

Open `http://127.0.0.1:8873` and choose a sentence.
Press play on a model card. Starting another player pauses the first.
Selecting another sentence stops the current recording.
