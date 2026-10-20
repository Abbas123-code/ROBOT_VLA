# CPU-only vision tools. Isaac Sim, CVAT and host environments are not included.
FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 MPLCONFIGDIR=/tmp/matplotlib HOME=/tmp
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY configs/requirements-yolov9-cpu.txt configs/requirements-yolov9-cpu.txt
RUN pip install --no-cache-dir torch==2.5.1+cpu torchvision==0.20.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r configs/requirements-yolov9-cpu.txt
COPY configs/yolov9_official.json configs/yolov9_official.json
COPY scripts/fetch_official_yolov9.py scripts/fetch_official_yolov9.py
RUN python scripts/fetch_official_yolov9.py \
    && git config --system --add safe.directory /app/third_party/yolov9
COPY src/perception/ src/perception/
COPY scripts/portable_vision.py scripts/portable_vision.py
RUN mkdir -p /input /output && chmod 1777 /output
USER 1000:1000
ENTRYPOINT ["python", "/app/scripts/portable_vision.py"]
CMD ["--help"]