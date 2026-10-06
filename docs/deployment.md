# Deployment preparation — not deployed

1. Resolve code licensing, attribution and weight/audio/text redistribution in
   `release-gaps.md`; decide which assets may become public.
2. Choose author-controlled external storage. Add only real approved HTTPS URLs,
   retaining manifest hashes. Do not commit weights to ordinary Git.
3. Validate the native frontend and dependencies on the target host. A Windows
   build does not certify Streamlit Cloud's Linux environment.
4. Provision approved artifacts, verify them, then perform functional smoke checks.
5. Set the entry point to `demo/streamlit_app.py` and install from `pyproject.toml`.
   A native toolchain/container or compliant frontend wheel may be needed. No
   public server is started by the preparation tools.

Local tests established HTTP health and UI/error-state behavior. Loaded-model
peak RAM, CPU saturation and concurrent-user capacity have **not** been measured.
Do not promise a RAM tier, free hosting capacity, or Intel N150/Pi 5 performance.
The FP32 graph files are about 61–65 MiB each; sessions and workspaces need additional
memory, currently unmeasured.

Settings: two ORT intra-op threads, one inter-op thread, one active synthesis per
process, 300 characters, 800 IDs and a 60-second accepted-output limit. That check
occurs after inference and is not a hard compute/memory sandbox. Process isolation,
timeouts, request admission and target-host capacity testing remain deployment work.
No ONNX quality equivalence, PTQ support or edge readiness is claimed.

Operator environment variables: `HIFIMOBINET_MODEL_DIR`, `HIFIMOBINET_AUDIO_DIR`,
and optionally `HIFIMOBINET_HOME`. Keep hosting secrets outside Git. No secret is
required for a local installation with local assets.
