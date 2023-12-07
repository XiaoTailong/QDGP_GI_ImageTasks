#!/bin/bash
#SBATCH --job-name=DGP
#SBATCH --partition=CLUSTER
#SBATCH -t 7-00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodelist=compute-0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -o ./output/%x-%j.out
#SBATCH -e ./output/%x-%j.err
module load python/anaconda3
module load cuda/cuda-11.4
source activate tf_tc

#IMAGE_PATH="/data/282_cat_7.jpeg"
#CLASS="282"

IMAGE_PATH="/data/bee2.JPEG"
CLASS="309"


python -u -W ignore example.py \
--image_path $IMAGE_PATH \
--class $CLASS \
--seed 4 \
--dgp_mode category_transfer \
--update_G \
--ftr_num 8 8 8 \
--ft_num 7 7 7 \
--lr_ratio 1 1 1 \
--w_D_loss 1 1 1 \
--w_nll 0.2 \
--w_mse 0 0 0 \
--select_num 0 \
--sample_std 0.5 \
--iterations 125 125 100 \
--G_lrs 2e-7 2e-5 2e-6 \
--z_lrs 1e-1 1e-2 2e-4 \
--use_in False False False \
--resolution 256 \
--weights_root pretrained \
--load_weights 256 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema