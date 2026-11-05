#!/usr/bin/env bash
# Sourced only inside the numbered child scripts, never in the user's terminal.
set -euo pipefail
DEMO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO_PYTHON="${HIFIMOBINET_N100_PYTHON:-$HOME/venvs/openvino-n100/bin/python}"
export PYTHONPATH="$DEMO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export NNCF_TELEMETRY_DISABLED=1
export OPENVINO_TELEMETRY_DISABLED=1
export HIFIMOBINET_N100_BUNDLE="${HIFIMOBINET_N100_BUNDLE:-$HOME/benchmarks/hifi-mobiNet-n100-openvino-2026-10-31/hifi-mobiNet-n100-openvino-2026-10-31}"
export HIFIMOBINET_N100_RUNS="${HIFIMOBINET_N100_RUNS:-$HOME/benchmarks/n100-demo-runs}"
cd -- "$DEMO_ROOT"
if [[ ! -x "$DEMO_PYTHON" ]]; then
    printf 'Khong tim thay Python N100: %s\n' "$DEMO_PYTHON" >&2
    exit 1
fi
if [[ -f SHA256SUMS ]]; then
    sha256sum --check --quiet SHA256SUMS
fi
mkdir -p -- "$HIFIMOBINET_N100_RUNS"
DEMO_LOG_DIR="$(mktemp -d "$HIFIMOBINET_N100_RUNS/$(date -u +%Y%m%dT%H%M%SZ)-${DEMO_STEP}.XXXXXX")"
printf 'Log: %s\n' "$DEMO_LOG_DIR"
