FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ libtcmalloc-minimal4 libnuma1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --default-timeout=300 --retries 10 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt \
 && (pip check || true)

RUN groupadd --gid 1000 vllm \
 && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin vllm \
 && mkdir -p /models \
 && chown vllm:vllm /models

COPY vllm_privacy /opt/vllm_privacy
COPY logging_config.json /opt/logging_config.json
COPY scripts /opt/scripts
ENV PYTHONPATH=/opt

ENV HF_HOME=/models \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

USER 1000:1000
WORKDIR /home/vllm
EXPOSE 8000
ENTRYPOINT ["vllm", "serve"]