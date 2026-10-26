FROM python:3.12-slim

ARG VLLM_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
        libtcmalloc-minimal4 libnuma1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_39_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
ENV VLLM_CPU_KVCACHE_SPACE=1

EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]