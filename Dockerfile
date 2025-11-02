FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

# Основные переменные окружения
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Установка системных пакетов
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev \
    git wget libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Обновление pip
RUN pip3 install --ignore-installed --upgrade pip setuptools wheel --break-system-packages
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 --break-system-packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt --break-system-packages


WORKDIR /workspace


COPY . /workspace

RUN mkdir -p /workspace/data/train /workspace/data/val /workspace/models /workspace/results

RUN chmod +x /workspace/scripts/*.sh 2>/dev/null || true

CMD ["/bin/bash"]