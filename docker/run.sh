#!/bin/bash
# Run this script from within the refacer/docker folder.
# You'll need inswrapper_128.onnx from either:
#    https://drive.google.com/file/d/1eu60OrRtn4WhKrzM4mQv4F3rIuyUXqfl/view?usp=drive_link
# or https://drive.google.com/file/d/1jbDUGrADco9A1MutWjO6d_1dwizh9w9P/view?usp=sharing
# or https://mega.nz/file/9l8mGDJA#FnPxHwpdhDovDo6OvbQjhHd2nDAk8_iVEgo3mpHLG6U
# or https://1drv.ms/u/s!AsHA3Xbnj6uAgxhb_tmQ7egHACOR?e=CPoThO
# or https://civitai.com/models/80324?modelVersionId=85159

docker stop -t 0 refacer
docker build -t refacer -f Dockerfile.nvidia . && \
    docker run --rm --name refacer -v $(pwd)/..:/refacer -p 7860:7860 --gpus all refacer python3 app.py --server_name 0.0.0.0 &
sleep 2 && google-chrome --new-window "http://127.0.0.1:7860" &
