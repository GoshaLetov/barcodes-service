#!/bin/bash

set -ue

docker run \
    --detach \
    --publish {{ service_port }}:5000 \
    --name {{ container_name }} \
    --restart always \
    {{ docker_image }}:{{ docker_tag }}
