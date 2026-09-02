FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-venv python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rlvr-posttraining
COPY . .
RUN python3 -m pip install --upgrade pip && python3 -m pip install -e ".[dev]"

CMD ["bash"]
