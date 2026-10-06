# Hugging Face preparation checkpoint

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
