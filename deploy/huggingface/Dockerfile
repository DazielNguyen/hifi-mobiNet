FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 appuser \
    && install -d -o appuser -g appuser /home/appuser/app
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HIFIMOBINET_ASSET_DIR=/home/appuser/assets \
    HIFIMOBINET_ENABLE_HARVARD_AUDIO=0
RUN python -m pip install --no-cache-dir \
    cmake==3.31.10 ninja==1.13.0 huggingface_hub==2.1.1 \
    onnxruntime==1.23.2 streamlit==1.50.0 numpy==2.4.6
WORKDIR /home/appuser/app
COPY --chown=appuser:appuser . .
RUN python -m pip install --no-cache-dir -e ".[demo]"
RUN python -m pip install --no-cache-dir ./vendor/banhmi-phonemize \
    --config-settings=build-dir=/home/appuser/app/.local/native-build \
    --config-settings=cmake.version="==3.31.10"
RUN python -m pip check
EXPOSE 8501
CMD ["sh", "deploy/huggingface/start.sh"]
