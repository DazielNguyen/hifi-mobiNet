#!/usr/bin/env bash
DEMO_STEP=check
source "$(dirname -- "$0")/common.sh"
"$DEMO_PYTHON" -m hifimobinet.n100_demo doctor --output "$DEMO_LOG_DIR/doctor.json" \
    2>&1 | tee "$DEMO_LOG_DIR/check.log"
printf '\nBUOC 02 OK. Tiep tuc buoc 03.\n'
