FROM nvidia/cuda:11.2.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    software-properties-common \
    wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    libcudnn8-dev \
    libopencv-dev \
    python3-dev \
    python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s pip3 /usr/bin/pip

# DarkNet
RUN cp /usr/local/cuda/compat/* /usr/local/cuda/targets/x86_64-linux/lib/
RUN git clone https://github.com/AlexeyAB/darknet.git
WORKDIR /darknet
COPY build.sh .
RUN chmod +x build.sh && ./build.sh
