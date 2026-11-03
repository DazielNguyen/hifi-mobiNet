#!/usr/bin/env bash
DEMO_STEP=server
source "$(dirname -- "$0")/common.sh"
printf '\nMo trinh duyet TREN PC: http://<IPv4-cua-N100>:8501\n'
printf 'Dia chi tren N100 (chon dia chi LAN cung mang voi PC):\n'
hostname -I
printf 'Giu terminal nay mo. Ctrl+C de dung demo.\n\n'
"$DEMO_PYTHON" -u -m hifimobinet.n100_server --host 0.0.0.0 --port 8501 \
    2>&1 | tee "$DEMO_LOG_DIR/server.log"
