# GPU image for the streaming pipeline: docker build -t anpr . && docker run --gpus all anpr
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY weights/manifest.json ./weights/manifest.json
RUN pip install --no-cache-dir -e . rfdetr
ENTRYPOINT ["python", "-m", "anpr.run"]
CMD ["--config", "configs/default.yaml"]
