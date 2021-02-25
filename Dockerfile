FROM nvidia/cuda:11.2.0-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    software-properties-common \
    wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    libcudnn8-dev \
    libopencv-dev \
    python3-dev \
    python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s pip3 /usr/bin/pip
RUN pip install opencv-python

# DarkNet
RUN cp /usr/local/cuda/compat/* /usr/local/cuda/targets/x86_64-linux/lib/
RUN git clone https://github.com/AlexeyAB/darknet.git
WORKDIR /darknet
COPY build.sh .
RUN ./build.sh

# Object Detection
COPY object_detection.py .
ENTRYPOINT ["./object_detection.py", "/net.cfg", "/net.data", "/net.weights", "/data/inputs", "--save_to", "/data/outputs"]

# RUN pip install gdown
# RUN gdown --id 1V726wzWW1iBDcJ7uP8c0n_Y4M98lIg-C --output net.cfg
# RUN gdown --id 1wK66ga9YgtjGNSm9fpouJn2GaDxa7SfT --output net.weights
# RUN ./darknet detector test ./cfg/coco.data net.cfg net.weights data/dog.jpg -i 0 -thresh 0.25
