#!/bin/bash

sudo docker run --shm-size=2g --privileged --gpus all -it --rm -p 8888:8888 -v `pwd`:/scratch --user $(id -u):$(id -g) --workdir=/scratch -e HOME=/scratch my-sg2 pipenv run jupyter notebook
