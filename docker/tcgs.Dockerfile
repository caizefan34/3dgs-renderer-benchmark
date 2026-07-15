FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG TCGS_REPO=https://github.com/DeepLink-org/3DGSTensorCore.git
ARG TCGS_COMMIT=0bb82f88fde211c34b42e1497f0fc7265461592b

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/3dgs-renderer-benchmark

RUN apt-get update && apt-get install -y --no-install-recommends \
    git python${PYTHON_VERSION} python3-pip python3-venv build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch --index-url ${TORCH_INDEX_URL}

RUN git clone ${TCGS_REPO} /opt/3DGSTensorCore \
    && cd /opt/3DGSTensorCore \
    && git checkout ${TCGS_COMMIT}

COPY requirements-test.txt ./
RUN python3 -m pip install -r requirements-test.txt

COPY . .

CMD ["python3", "src/run_benchmark.py", "--list-renderers"]

