#!/bin/sh
set -eu
python deploy/huggingface/bootstrap.py
exec python -m streamlit run demo/streamlit_app.py \
  --server.address 0.0.0.0 --server.port 8501 \
  --server.headless true --browser.gatherUsageStats false
