# darknet-docker
Run darknet in docker.

## Usage

1. Install the prerequisites
2. Build the docker container
3. Apply darknet inside a docker container

### Prerequisites
- [nividia-docker](https://github.com/NVIDIA/nvidia-docker)
- 

### Building the docker Image
When building the docker image we'll do the following
1. Pull the latest [nvidia CUDA docker](https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/11.2.0/ubuntu20.04-x86_64/devel/Dockerfile) for Ubuntu 20.04
2. Install build tools & dependencies (cmake, git, cudnn, python, ...)
3. Pull and build [darknet](https://github.com/AlexeyAB/darknet)
4. Set object detection entry point


### Applications

#### Object Detection
You can either call

```bash
docker run -t --runtime=nvidia -v <path/to/net.cfg>:/net.cfg -v <path/to/net.data>:/net.data -v <path/to/net.weights>:/net.weights -v <path/to/inputs>:/data/inputs -v <path/to/outputs>:/data/outputs darknet-docker:latest <more optional args>
```
directly, or you use `object_detection_run.sh` by calling
```bash
object_detection_run.sh <path/to/net.cfg> <path/to/net.data> <path/to/net.weights> <path/to/inputs> <path/to/outputs>
```



## To Do's

- [ ] Multiple Dockerfiles
- [ ] Host Dockerfiles on docker hub
- [ ] Add classification example
