#!/usr/bin/env bash
# Run with bash, never source this into an interactive terminal.
DEMO_STEP=common97
source "$(dirname -- "$0")/common.sh"
export NNCF_TELEMETRY_DISABLED=1
export OPENVINO_TELEMETRY_DISABLED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
if [[ -f "$DEMO_ROOT/.local/n100-common97-models.json" ]]; then
    printf 'common97 da co catalog; kiem tra lai, khong ghi de.\n'
    "$DEMO_PYTHON" -m hifimobinet.n100_demo doctor
    printf 'BUOC COMMON97 OK: catalog hien co da vuot kiem tra.\n'
    exit 0
fi
COMMON97_WORK="$HIFIMOBINET_N100_BUNDLE/work-common97-$(date -u +%Y%m%dT%H%M%SZ)"
"$DEMO_PYTHON" -u scripts/n100_demo/prepare_common97.py --work "$COMMON97_WORK" \
    2>&1 | tee "$DEMO_LOG_DIR/prepare-common97.log"
printf '\nBUOC COMMON97 OK. Chay 02-check.sh, sau do 03-start.sh.\n'
