FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/3dgs-renderer-benchmark

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python${PYTHON_VERSION} python3-pip python3-venv build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-test.txt ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch --index-url ${TORCH_INDEX_URL} \
    && python3 -m pip install -r requirements-test.txt \
    && python3 -m pip install gsplat

COPY . .

CMD ["python3", "src/run_benchmark.py", "--list-renderers"]

