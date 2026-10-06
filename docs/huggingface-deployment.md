# Hugging Face handoff for DazielNguyen

This preparation is local. It does not create repositories, authenticate, upload, push, release or deploy.
The namespace comes from the author. The targets below are planned repository IDs, not confirmed public URLs.

| Type | Planned ID | Local candidate |
|---|---|---|
| Model | `DazielNguyen/hifi-mobiNet` | `model/` |
| Dataset | `DazielNguyen/hifi-mobiNet-harvard` | `dataset/` |
| Docker Space | `DazielNguyen/hifi-mobiNet` | `space/` |

## Local candidates

The candidates are outside Git at `~/hifi-mobiNet-hf-staging-v0.1.0/`.
Each candidate has a file inventory in `release-manifest.json`. It excludes itself to avoid circular hashes.
`PREPARATION.json` records package counts. Checkpoints, ONNX files, WAVs and publication-intended log/config copies retain their supplied bytes.
The source asset directory and original transfer ZIPs remain unchanged. No checkpoint deserialization or weights-only conversion occurs.
The model candidate includes four selected checkpoints, four ONNX graphs, configs, redacted logs, notices and attribution.
The dataset candidate includes all 2,880 WAVs and the original alignment catalog. Sentence distribution rights remain pending confirmation.
The Space candidate contains allowlisted code and source for the native frontend. It contains no model, audio, environment or token files.

To create another candidate from verified assets, use a new destination:

```sh
cd /Users/vananhduy/Documents/Repository_Git_Hub/hifi-mobiNet
source .venv/bin/activate
python scripts/huggingface/prepare_release.py \
  --assets "$HOME/hifi-mobiNet-assets-v0.1.0" \
  --destination "$HOME/hifi-mobiNet-hf-staging-next"
```

The script copies Git-listed working files. The package inventory records actual bytes, including any staged changes before commit.
The recorded Git HEAD is the preparation base, not proof that every candidate file belongs to that commit.
Existing destinations and source/destination symlinks are rejected. No network call forms part of candidate preparation.

## Account and initial repositories

Install the pinned Hub client in the local environment:

```sh
python -m pip install huggingface_hub==2.1.1
hf auth login
hf auth whoami
```

Use the browser or terminal login flow. Keep tokens outside Git and chat.
The following repository creation and upload commands are future actions. This preparation does not run them.
Create the model and dataset as **private** initially. Create the Space as private with the Docker SDK.
Use the [repository creation UI](https://huggingface.co/new) and [Space creation UI](https://huggingface.co/new-space).
The current [Spaces documentation](https://huggingface.co/docs/hub/spaces-overview) describes account eligibility and hardware costs.
Check the account plan before selecting compute. No hardware purchase forms part of this preparation.

## Upload candidates after review

Review checkpoint embedded metadata, remaining component conditions and the file inventories before upload.
Review the private dataset candidate separately. The audio license does not cover the Harvard sentence text.
The original transfer ZIPs keep their earlier license-pending records. These candidates include the subsequent scoped decisions.

Once the private repositories exist, these commands upload each candidate with a separate commit:

```sh
hf upload DazielNguyen/hifi-mobiNet \
  "$HOME/hifi-mobiNet-hf-staging-v0.1.0/model" . \
  --commit-message "Add identified model artifacts and scoped license records"

hf upload DazielNguyen/hifi-mobiNet-harvard \
  "$HOME/hifi-mobiNet-hf-staging-v0.1.0/dataset" . --repo-type dataset \
  --commit-message "Add private historical audio candidate and attribution"
```

Private visibility is a repository setting, not an upload-command safeguard. Verify the setting before upload.
Do not upload the entire staging parent, transfer folder, virtual environment or source asset directory.
The official [CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) explains repository types and commit messages.

## Configure the Docker Space

The [Docker SDK](https://huggingface.co/docs/hub/spaces-sdks-docker) supports the existing Streamlit app.
The generated Space README sets `sdk: docker` and `app_port: 8501`.
The startup binds Streamlit to `0.0.0.0:8501`. The container runs as UID 1000.
The image builds the unchanged native frontend from its pinned eSpeak NG source. It retains source and license notices.
The image installs Python 3.12, ONNX Runtime 1.23.2, Streamlit 1.50.0 and Hub client 2.1.1.
These pins do not fix every transitive dependency or the base-image digest. The actual validation record identifies the tested image.

Obtain the full commit hash from the uploaded model repository. Set this Space variable:

| Variable | Value |
|---|---|
| `HIFIMOBINET_MODEL_REVISION` | Actual full 40-character commit from the uploaded model repository |
| `HIFIMOBINET_ENABLE_HARVARD_AUDIO` | `0` initially |

For private model assets, add a read-only `HF_TOKEN` in **Space Settings > Secrets**.
The token must permit access to the private model repository. Do not use a write token for runtime downloads.
The startup downloads only four ONNX graphs. It verifies their recorded sizes and SHA-256 values.
The Space never downloads training checkpoints or loads pickle files. A missing revision or mismatched file stops startup.
The runtime cache is temporary. Startup can fetch the pinned assets again after a restart.
The official [download guide](https://huggingface.co/docs/huggingface_hub/guides/download) explains full commit revisions and caching.

After resolving the relevant Harvard distribution conditions, configure these additional variables for historical comparison:

| Variable | Value |
|---|---|
| `HIFIMOBINET_ENABLE_HARVARD_AUDIO` | `1` |
| `HIFIMOBINET_AUDIO_REVISION` | Actual full 40-character commit from the dataset repository |

The operator setting does not certify distribution rights. Keep comparison disabled until the appropriate review is complete.
A private dataset requires token read access. The downloaded catalog must match its pinned hash before its WAV paths are used.
The runtime verifies all 2,880 WAV sizes and hashes. Source text and alignment remain distinct from audio license scope.

Upload the prepared Space only after inspecting its candidate:

```sh
hf upload DazielNguyen/hifi-mobiNet \
  "$HOME/hifi-mobiNet-hf-staging-v0.1.0/space" . --repo-type space \
  --commit-message "Deploy reviewed Docker Streamlit candidate"
```

An upload to the Space triggers a build and runtime start. The local preparation does not perform this command.

## Local Linux validation

Build the candidate for Linux amd64:

```sh
docker build --platform linux/amd64 -t hifi-mobinet:hf-local \
  "$HOME/hifi-mobiNet-hf-staging-v0.1.0/space"
```

The image requires a model revision for normal startup. Remote downloading cannot be verified before the model repository is uploaded.
For local functional checks, mount the already verified asset directory read-only and bypass the remote startup:

```sh
docker run --rm --platform linux/amd64 \
  -v "$HOME/hifi-mobiNet-assets-v0.1.0:/assets:ro" \
  -e HIFIMOBINET_ASSET_DIR=/assets \
  --entrypoint python hifi-mobinet:hf-local \
  -m hifimobinet.cli verify --model sequential-ir
```

The runtime mode uses only exact local identities. Local Docker checks do not establish hosted performance or remote download behavior.

## Before public access

Check Space build/runtime logs, model availability, short synthesis and WAV playback.
Check memory use and concurrent requests on the actual target hardware. CPU Basic resources do not establish capacity for this app.
The existing limit serializes synthesis and caps input length. The output-length check is not a hard timeout or process sandbox.
Historical scores do not describe this hosted ONNX runtime. No benchmark or research metric belongs to deployment smoke checks.
Keep model, dataset and Space visibility decisions separate. Record actual repository revisions and URLs only after publication.
See `artifact-licensing.md` for scope and `huggingface-validation.json` for performed checks and remaining limitations.
