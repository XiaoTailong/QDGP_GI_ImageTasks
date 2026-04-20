#!/bin/bash

# 1. 确保 Python 能找到项目根目录下的 models 文件夹
export PYTHONPATH=$PYTHONPATH:.

# 2. 定义参数（你可以根据需要修改这里）
IMG_PATH="./data/cat1.JPEG"
SAVE_PATH="./results/inpainting_test"
WEIGHTS="./pretrained"

# 3. 创建结果目录
mkdir -p $SAVE_PATH

# 4. 执行核心程序
# 注意：我这里用了 python 而不是原始脚本中可能存在的交互式命令
# 并且手动拼接了必要的 flags
python dgp_qp_example_001.py \
    --image_path "$IMG_PATH" \
    --exp_path "$SAVE_PATH" \
    --dgp_mode inpainting \
    --seed 1 \
    --update_G \
    --update_embed \
    --iterations 200 \
    --resolution 256 \
    --weights_root "$WEIGHTS" \
    --load_weights 256
