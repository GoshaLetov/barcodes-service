#!/bin/bash

set -ue

docker stop {{ container_name }} || true

docker rm -f {{ container_name }} || true
