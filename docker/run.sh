#!/bin/bash
# Run this script from within the refacer/docker folder.

docker build -t refacer -f Dockerfile.nvidia . && \
    docker run --rm -v $(pwd)/out:/refacer/out -p 7860:7860 --gpus all refacer python3 app.py --server_name 0.0.0.0 && \
    sleep 1 && google-chrome --new-window "http://127.0.0.1:7860" &
