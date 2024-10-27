FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      fonts-dejavu-core \
      rsync \
      git \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      bc \
      aria2 \
      libgoogle-perftools4 \
      libtcmalloc-minimal4 \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Clone the A1111 repo
RUN git clone --depth=1 https://github.com/antonioglass/stable-diffusion-webui.git

# Install Python packages
RUN pip install --no-cache-dir torch==2.1.2+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

# Install A1111 Web UI
WORKDIR /stable-diffusion-webui
COPY install-automatic.py ./
RUN pip install -r requirements_versions.txt && \
    python -m install-automatic --skip-torch-cuda-test

# Cloning ControlNet extension repo
RUN git clone --depth=1 https://github.com/Mikubill/sd-webui-controlnet.git extensions/sd-webui-controlnet

# Cloning the ReActor extension repo
RUN git clone --depth=1 https://github.com/Gourieff/sd-webui-reactor.git extensions/sd-webui-reactor

# Cloning a person mask generator extension repo
RUN git clone --depth=1 https://github.com/djbielejeski/a-person-mask-generator.git extensions/a-person-mask-generator

# Installing dependencies for ReActor
WORKDIR /stable-diffusion-webui/extensions/sd-webui-reactor
RUN pip install protobuf==3.20.3 mediapipe==0.10.11 onnxruntime-gpu==1.16.3 && \
    pip install -r requirements.txt

# Configuring ReActor to use the GPU instead of CPU
RUN echo "CUDA" > last_device.txt

# "Installing dependencies for a person mask generator extension
WORKDIR /stable-diffusion-webui/extensions/a-person-mask-generator
RUN pip install -r requirements.txt

# Installing dependencies for ControlNet
WORKDIR /stable-diffusion-webui/extensions/sd-webui-controlnet
RUN pip install -r requirements.txt

# Installing the models for ReActor
WORKDIR /stable-diffusion-webui/models/insightface
RUN wget https://huggingface.co/antonioglass/reactor/resolve/main/inswapper_128.onnx

WORKDIR /stable-diffusion-webui/models/insightface/models/buffalo_l
RUN wget https://huggingface.co/antonioglass/reactor/resolve/main/buffalo_l/1k3d68.onnx && \
    wget https://huggingface.co/antonioglass/reactor/resolve/main/buffalo_l/2d106det.onnx && \
    wget https://huggingface.co/antonioglass/reactor/resolve/main/buffalo_l/det_10g.onnx && \
    wget https://huggingface.co/antonioglass/reactor/resolve/main/buffalo_l/genderage.onnx && \
    wget https://huggingface.co/antonioglass/reactor/resolve/main/buffalo_l/w600k_r50.onnx

# Installing Codeformer
WORKDIR /stable-diffusion-webui/models/Codeformer
RUN wget https://huggingface.co/antonioglass/reactor/resolve/main/codeformer-v0.1.0.pth

WORKDIR /stable-diffusion-webui/models/GFPGAN
RUN wget https://huggingface.co/antonioglass/reactor/resolve/main/detection_Resnet50_Final.pth && \
    wget https://huggingface.co/antonioglass/reactor/resolve/main/parsing_parsenet.pth

# Download a person mask generator model
WORKDIR /stable-diffusion-webui/models/mediapipe
RUN wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite

# Download ControlNet models
WORKDIR /stable-diffusion-webui/models/ControlNet
RUN wget https://huggingface.co/antonioglass/controlnet/raw/main/controlnet11Models_openpose.yaml && \
    wget https://huggingface.co/antonioglass/controlnet/resolve/main/controlnet11Models_openpose.safetensors

# Download VAEApprox model
WORKDIR /stable-diffusion-webui/models/VAE-approx
RUN wget https://huggingface.co/antonioglass/models/resolve/main/vaeapprox-sdxl.pt

# Create log directory
WORKDIR /logs

# Install config files
WORKDIR /stable-diffusion-webui
RUN rm -f webui-user.sh config.json ui-config.json
COPY webui-user.sh config.json ui-config.json ./

# Install Worker dependencies
RUN pip install requests runpod==1.6.2 huggingface_hub

# Add RunPod Handler and Docker container start script
WORKDIR /
COPY start.sh rp_handler.py ./
COPY schemas /schemas

# Start the container
RUN chmod +x /start.sh
ENTRYPOINT /start.sh
