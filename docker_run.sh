#!/bin/bash

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

set -e

# Wrapper script for setting up `docker run` to properly
# cache downloaded files, custom extension builds and
# mount the source directory into the container and make it
# run as non-root user.
#
# Use it like:
#
# ./docker_run.sh python generate.py --help
#
# To override the default `stylegan2ada:latest` image, run:
#
# IMAGE=my_image:v1.0 ./docker_run.sh python generate.py --help
#

rest=$@

IMAGE="${IMAGE:-sg2ada:latest}"

CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${IMAGE} 2> /dev/null)
if [[ "${CONTAINER_ID}" ]]; then
    docker run --shm-size=2g --gpus all -it --rm -v `pwd`:/scratch --user $(id -u):$(id -g) \
        --workdir=/scratch -e HOME=/scratch $IMAGE $@
else
    echo "Unknown container image: ${IMAGE}"
    exit 1
fi
