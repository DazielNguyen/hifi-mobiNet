# Model access

`manifest.json` is the only model allowlist. It records four selected checkpoints
and their later Q05 FP32 ONNX exports. **Weights are not committed.** An artifact
hash proves identity, not its license or the identity of every historical run.

Install an authorized copy of each ONNX into `models/weights/`, using the exact
filename and SHA-256 in the manifest. Alternatively, an administrator can set
`HIFIMOBINET_MODEL_DIR` to an external directory. End users of the web demo cannot
select arbitrary paths or upload checkpoints. Run:

```sh
python -m hifimobinet.cli models
python -m hifimobinet.cli verify --model sequential-ir
```

The team ONNX/checkpoint URLs are pending. The existing upstream Piper training
checkpoint URL is recorded from its previous download manifest, with its original
SHA-256. To explicitly fetch that approximately 807 MiB training file:

```sh
python scripts/inference/download_model.py --model piper-original --kind checkpoint --confirm-large-download
```

This is a **training checkpoint**, not the Q05 ONNX required by the demo. The
download command does not convert or load it. The repository never uses unsafe
pickle loading. New export/quantization is outside this task; no automated
checkpoint-to-ONNX fallback is provided.

Model compatibility and architecture fields are in `configs/*.json`; original
templates remain under `configs/original/`. Runtime scales remain
`[0.667, 1.0, 0.8]`, output is mono 22,050 Hz, and only CPU FP32 ONNX is enabled
for functional validation. Historical INT8 graphs are excluded from the demo.

The author confirmed sharing permission on 2026-10-06. Specific licenses, external
storage and public URLs remain unconfirmed. The Piper card's dataset license is
not a blanket weight license. All four existing Q05 graphs passed a short local
functional synthesis check; these passes do not bind historical quality scores to
the new runtime. Selected training checkpoints total 3.44 GB and include training
state; no weights-only conversion was performed. See `docs/artifact-inventory.json`.
