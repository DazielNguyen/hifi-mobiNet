# Hugging Face checkpoint

## Current state: fixed listening demo

Date: 2026-10-06. The separate Static Space is public at
https://huggingface.co/spaces/DazielNguyen/hifi-mobiNet.
It contains twelve new demo WAVs, three sentences and four models.
Audio URLs are pinned to the verified asset revision.
GitHub Pages deployment and browser checks are recorded in listening-publication.json.
Original checkpoints remain private; Harvard audio and sentence text are excluded.
No paid plan, compute instance or live inference host was requested.
See listening-demo.md for provenance, scoped audio terms and maintenance.

Earlier sections preserve prior preparation states.

## Current state: public inference, private checkpoints

Date: 2026-10-06. Public inference URL: https://huggingface.co/DazielNguyen/hifi-mobiNet-inference.
Public model revision: `f760e85a45b84c217091acf963837ffc817bad8d`.
Four existing ONNX graphs passed anonymous pinned downloads and original SHA-256 checks.
The checkpoint archive `DazielNguyen/hifi-mobiNet` remains private. Original path-bearing checkpoint bytes were not rewritten.
See huggingface-inference-release.md/json and model-publication-review.md for completed checks and scope.
The source runtime target now points to the public inference repository. The older prepared Space snapshot keeps its original target.
Regenerate a Space candidate from current committed code before future deployment.
Full model candidates now target the separate private archive. A configuration that points them at the inference repository is rejected.
Dataset upload and demo hosting remain deferred. No GitHub push or paid compute occurred.

Next: use the public download guide, then choose demo hosting.
Historical audio stays off while Harvard text distribution conditions remain unresolved.

## Previous checkpoint

## Latest state: private model uploaded

Date: 2026-10-06. The author chose model upload first and deferred hosting.
Account: DazielNguyen, authenticated; no PRO plan at the check.
Model URL: https://huggingface.co/DazielNguyen/hifi-mobiNet. Visibility: private.
Current verified model commit: `a4354007630f14b2cd0046a6f6b0dc17dd1f7ac9`.
See huggingface-model-upload.md and huggingface-model-upload.json for the completed checks and download record.
The dataset and Space candidates remain local. Public release, demo deployment and GitHub push remain undone.

Next: finish the public artifact review, decide model visibility and choose demo hosting.
Use the exact model commit for downloads. Do not replace it with `main` in runtime configuration.
Keep historical audio off while Harvard text distribution conditions remain unresolved.

## Historical local preparation

Date: 2026-10-06. Namespace: DazielNguyen. Prepared code commit: `53bb061cd3ec98ff0d322b31e6e808be150e02de`.
The three candidates are local at `~/hifi-mobiNet-hf-staging-v0.1.0/`.
No Hugging Face repository, tag, release or deployment was created. No token was accessed or stored by this preparation.

## Verified

- Selected model, audio, config and log copies match supplied identities.
- The candidate inventories bind the current files. Source artifacts and the 121 imported source mappings remain unchanged.
- Twenty related host tests passed across the recorded runs.
- Linux amd64 Docker build and dependency checks passed. Four identified ONNX models synthesized valid mono PCM16 at 22,050 Hz.
- Three saved frontend ID sequences matched. All 2,880 historical WAV hashes and recorded PCM headers passed verification.
- Streamlit AppTest showed 720 choices, four players for three selected sentences and successful Sequential-IR synthesis.
- Offline TTS-only bootstrap verified four models. The local HTTP health check returned 200. The temporary test container was stopped.

## Continue

1. Install the pinned Hub client and log in using the commands in `huggingface-deployment.md`.
2. Review candidate inventories and remaining component/checkpoint metadata conditions.
3. Create the planned repositories privately and upload reviewed candidates only when authorized.
4. Record actual HF commits and configure `HIFIMOBINET_MODEL_REVISION`. A private model repository needs a read-only Space secret.
5. Verify the hosted startup, TTS and resource limits. Keep historical comparison off while Harvard text distribution conditions remain unresolved.

Local emulation does not establish hosted performance, remote download behavior or human listening results.
Full software versions, image identity and tested runtime hashes appear in `huggingface-validation.json`.
