#!/usr/bin/env bash
# Create the training environment for a clean checkout on Ubuntu/WSL2.
#
#   export HIFI_TRAIN_ROOT=/path/on/linux/filesystem/hifimobinet-train
#   export PYTHON310=/path/to/python3.10        # base interpreter, not modified
#   bash scripts/training/setup_ubuntu.sh
#
# Creates $HIFI_TRAIN_ROOT/venv (never touches other environments), installs the
# pinned requirements, installs this checkout in editable mode without
# re-resolving dependencies, builds both MAS extensions and writes
# $HIFI_TRAIN_ROOT/environment.json. Re-running reuses an existing venv.
set -euo pipefail

: "${HIFI_TRAIN_ROOT:?set HIFI_TRAIN_ROOT to a directory on the Linux filesystem}"
PYTHON310="${PYTHON310:-python3.10}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$HIFI_TRAIN_ROOT/venv"

case "$REPO" in /mnt/*) echo "warning: checkout is on a Windows drive; clone it onto the Linux filesystem for training I/O" >&2;; esac
"$PYTHON310" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' \
    || { echo "PYTHON310 must be a Python 3.10 interpreter" >&2; exit 1; }

mkdir -p "$HIFI_TRAIN_ROOT"
[ -x "$VENV/bin/python" ] || "$PYTHON310" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet pip==24.0
"$VENV/bin/python" -m pip install -r "$REPO/requirements/training-linux-cu130.txt"
"$VENV/bin/python" -m pip install --no-deps -e "$REPO"
bash "$REPO/scripts/training/build_mas.sh" "$VENV/bin/python"
"$VENV/bin/python" -m hifimobinet.training.env > "$HIFI_TRAIN_ROOT/environment.json"
"$VENV/bin/python" -m pip freeze > "$HIFI_TRAIN_ROOT/pip-freeze.txt"
echo "environment recorded in $HIFI_TRAIN_ROOT/environment.json"
