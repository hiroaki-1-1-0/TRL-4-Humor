FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs && \
    rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONPATH=/workspace \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9"

CMD ["bash"]
