# GPU Dockerfile for zk0 SuperExec (LeRobot v0.3.3 compatible)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PATH=/lerobot/.venv/bin:$PATH

# Preseed tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system deps + uv (root)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    build-essential git curl libglib2.0-0 libegl1-mesa ffmpeg libusb-1.0-0-dev \
    libopencv-dev python3-opencv \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && useradd --create-home --shell /bin/bash user_lerobot \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /lerobot
RUN chown -R user_lerobot:user_lerobot /lerobot

USER user_lerobot

ENV HOME=/home/user_lerobot \
    HF_HOME=/home/user_lerobot/.cache/huggingface \
    HF_LEROBOT_HOME=/home/user_lerobot/.cache/huggingface/lerobot \
    TORCH_HOME=/home/user_lerobot/.cache/torch

RUN uv venv

# LeRobot v0.3.3 + flwr + zk0
COPY pyproject.toml README.md ./
COPY src/ src/
RUN uv pip install --no-cache ".[all]" "lerobot[smolvla]==0.3.3" "flwr[superexec]==1.23.0" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY . .

CMD ["/bin/bash"]
ENTRYPOINT ["flower-superexec"]