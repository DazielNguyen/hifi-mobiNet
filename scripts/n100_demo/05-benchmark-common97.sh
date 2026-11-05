#!/usr/bin/env bash
DEMO_STEP=benchmark-common97
source "$(dirname -- "$0")/common.sh"
COMMON97_WORK="$($DEMO_PYTHON -c 'from hifimobinet.n100_demo import read,CATALOG,bundle_path; print(bundle_path()/read(CATALOG)["common97"]["work"])')"
"$DEMO_PYTHON" -u scripts/evaluation/edge/openvino_suite.py run \
    --bundle "$HIFIMOBINET_N100_BUNDLE" --work "$COMMON97_WORK" \
    --output "$DEMO_LOG_DIR/results" --threads 4 --rounds 3 \
    2>&1 | tee "$DEMO_LOG_DIR/benchmark.log"
printf '\nBENCHMARK COMMON97 OK: %s/results\n' "$DEMO_LOG_DIR"
