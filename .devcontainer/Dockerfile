FROM nvidia/cuda:13.0.1-devel-ubuntu22.04

LABEL maintainer="Elvis" description="Restormer CVPR2022 - Blackwell sm_120"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common build-essential gcc g++ wget curl git \
        libgl1 libglib2.0-0 libgl1-mesa-glx ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-distutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -q https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && rm get-pip.py && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# PyTorch 2.12 (sm_120 support)
RUN pip install --no-cache-dir torch==2.12.0 torchvision
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Restormer dependencies
RUN pip install --no-cache-dir \
    matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm \
    einops gdown addict future lmdb numpy pyyaml requests scipy tensorboard yapf lpips \
    pillow ninja packaging

WORKDIR /workspace
CMD ["bash"]
