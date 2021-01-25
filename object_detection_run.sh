#!/usr/bin/sh
docker run -t --runtime=nvidia -v $1:/net.cfg -v $2:/net.data -v $3:/net.weights -v $4:/data/inputs -v $5:/data/outputs darknet-docker:latest