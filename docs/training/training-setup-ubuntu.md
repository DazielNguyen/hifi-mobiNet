# Training setup on Ubuntu / WSL2

Validated on Ubuntu 24.04.4 LTS under WSL2 (distro name `Ubuntu-24.04`) with two
RTX 4070 Ti (12 GiB each), driver 596.36, 62 GiB RAM. Exact observed versions:
[environment-record.json](environment-record.json).

## Storage layout

Keep everything large on the Linux filesystem, outside Git and OneDrive:

```text
$HIFI_TRAIN_ROOT/
  venv/          training interpreter (created by the setup script)
  checkout/      a clone of this repository (recommended for I/O speed)
  data/          small derived datasets (e.g. smoke subsets)
  runs/          one directory per run: checkpoints, TensorBoard logs, records
```

The preprocessed LJSpeech dataset (`dataset.jsonl`, `config.json`, cached
tensors, about 22 GB) is read in place and never modified. Its `dataset.jsonl`
stores tensor paths relative to the directory it was produced from; pass that
directory as `--data-root`.

| Variable | Meaning |
|---|---|
| `HIFI_TRAIN_ROOT` | Linux-filesystem root for venv, data subsets and runs |
| `PYTHON310` | Base Python 3.10 interpreter used only to create the venv |
| `BANHMI_DATA_ROOT` | Directory the preprocessed dataset paths are relative to |
| `DATASET_DIR` | Directory containing `dataset.jsonl` and `config.json` |

## 1. Clone and create the environment

```sh
git clone <repository-url> "$HIFI_TRAIN_ROOT/checkout"
cd "$HIFI_TRAIN_ROOT/checkout"
bash scripts/training/setup_ubuntu.sh
```

The script creates `$HIFI_TRAIN_ROOT/venv`, installs pip 24.0 and
[requirements/training-linux-cu130.txt](../../requirements/training-linux-cu130.txt)
(torch 2.13.0+cu130, pytorch-lightning 1.7.7, ...), installs the checkout in
editable mode without re-resolving dependencies, builds both MAS extensions
(`scripts/training/build_mas.sh`) and writes `environment.json` and
`pip-freeze.txt` into `$HIFI_TRAIN_ROOT`.

Requirements: a C compiler (`build-essential`), an NVIDIA driver with CUDA 13
support, and a Python 3.10 interpreter. Training is supported from an editable
checkout only (`vendor/` is not packaged).

Why pip 24.0: pytorch-lightning 1.7.x declares `torch>=1.9.*`, which pip ≥ 24.1
rejects as invalid metadata. Why these versions: they are the versions present
in the existing workstation training interpreter, not the newest releases.
Upstream Piper pins `torch<2`; see the scheduler note in
[piper-component-comparison.md](piper-component-comparison.md) section 5.

## 2. Check the installation

```sh
"$HIFI_TRAIN_ROOT/venv/bin/python" -m pip install pytest==8.4.2   # test tool only
"$HIFI_TRAIN_ROOT/venv/bin/python" -m pytest tests/test_training.py -q
```

All tests must pass. Tests that need Lightning or the compiled MAS skip with an
explicit reason when those are missing, which then means the setup is incomplete.

## 3. Verify the dataset IDs (optional, needs the native frontend)

```sh
python scripts/training/verify_phoneme_ids.py --dataset "$DATASET_DIR/dataset.jsonl"
```

Re-phonemizes every text with `banhmi_phonemize` and compares IDs. On the
workstation this was run with the Windows build of the frontend: 13,100/13,100
rows matched. The Linux frontend build remains unverified (docs/setup.md).

## 4. Smoke test (short, real training)

```sh
PY="$HIFI_TRAIN_ROOT/venv/bin/python"
"$PY" - <<'EOF'
from pathlib import Path
import os
from hifimobinet.training.data import load_split, make_subset_dataset
split = load_split(Path("results/manifests/canonical_split.json"))
make_subset_dataset(Path(os.environ["DATASET_DIR"]), Path(os.environ["BANHMI_DATA_ROOT"]), split,
                    Path(os.environ["HIFI_TRAIN_ROOT"]) / "data/smoke-ljs-16train-4val", 16, 4)
EOF
D="$HIFI_TRAIN_ROOT/data/smoke-ljs-16train-4val"
for spec in "baseline-resblock2-vits2 baseline-resblock2-vits2" "Piper_no_VITS2_cpn piper-no-vits2-cpn"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=0 "$PY" -m hifimobinet.training.smoke --model-id "$1" \
    --config "configs/training/$2.yaml" --dataset-dir "$D" --data-root "$D" --split "$D/split.json" \
    --run-dir "$HIFI_TRAIN_ROOT/runs/smoke-$2" --batch-size 4 --epochs-a 1 --epochs-b 2 --max-updates 26
done
```

The subset takes the first 16 canonical-train and 4 canonical-validation
utterances; no test utterance is copied. Each smoke run writes
`smoke_report.json` (pass / fail / not_checked / not_applicable) and one
PyTorch inference WAV. Run models one at a time. Results:
[training-smoke-report.md](training-smoke-report.md).

## 5. Full training

See [next-run.md](next-run.md). Do not start a long run before the protocol in
[piper-no-vits2-protocol.md](piper-no-vits2-protocol.md) is settled.

## Known setup issues

| Symptom | Root cause | Fix in this repository |
|---|---|---|
| `MisconfigurationException: ... ExponentialLR doesn't follow PyTorch's LRScheduler API` | Lightning 1.7.7 checks `torch.optim.lr_scheduler._LRScheduler`; torch ≥ 2.0 schedulers derive from `LRScheduler` | `lr_scheduler_step` override with Lightning's default body (`training/harness.py`) |
| `UnpicklingError: Weights only load failed ... numpy.core.multiarray._reconstruct` on resume | torch ≥ 2.6 defaults to `weights_only=True`; a NumPy RNG state array was stored in the checkpoint | RNG state stored as tensors/plain values |
| `RuntimeError: cuFFT doesn't support tensor of type: BFloat16` with rhasspy Piper code | rhasspy/piper 73c04d8 computes the generator-output STFT under autocast | The model uses EdgeTTS's training step, which computes mel/STFT outside autocast (bf16 works) |
| `KeyError: 'audio_f0_path'` if EdgeTTS's own dataset class is used | EdgeTTS's `PiperDataset` requires F0 paths; this dataset has none | The harness reads rows like the baseline and passes `f0s=None` (Config A never uses F0) |
| Background install stops when the WSL shell exits | `nohup ... &` inside `wsl -e bash -c` is killed with the session | Run long commands in the foreground of a persistent terminal (or `tmux`) |
| `git clone` refuses: "dubious ownership" | The Windows checkout's `.git` belongs to another Windows account | Use `git -c safe.directory=<path> clone ...` for that command; global Git config left unchanged |
