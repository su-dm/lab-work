FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install packaging
RUN pip3 uninstall -y torch torchvision torchaudio
RUN pip3 install  torch torchvision torchaudio

COPY src/requirements.txt /opt/
RUN pip3 install -r /opt/requirements.txt -t /opt/lib
RUN pip3 install --upgrade datasets transformers trl
RUN pip3 install --upgrade --force-reinstall setuptools

# Copy the train directory and set permissions
COPY src /opt/src

# Set environment variable
ENV PYTHONPATH=/opt/lib

# Set working directory and entrypoint
WORKDIR /opt/src
ENTRYPOINT ["python3", "/opt/src/train.py"]
