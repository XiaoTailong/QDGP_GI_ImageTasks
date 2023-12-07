#!/bin/bash
#SBATCH --job-name=DGP_qp_n
#SBATCH --partition=CLUSTER
#SBATCH -t 7-00:00

#SBATCH --nodelist=compute-0-2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH -o ./output/%x-%j.out
#SBATCH -e ./output/%x-%j.err

module load python/anaconda3
module load cuda/cuda-11.4
source activate tf_tc

#IMAGE_PATH=data/ILSVRC2012_val_00004291.JPEG
#CLASS=442

#IMAGE_PATH="/data/388_pandas_9.jpeg"
#CLASS="388"
#IMAGE_PATH="/data/bee2.JPEG"
#CLASS="309"

#IMAGE_PATH="/data/ILSVRC2012_val_00003004.JPEG"
#CLASS="693"

#IMAGE_PATH="/data/161_dog_12.jpeg"
#CLASS="161"

#IMAGE_PATH="/data/ILSVRC2012_val_00044065.JPEG"
#CLASS="425"
IMAGE_PATH="/data/91_bird_15.jpeg"
CLASS="91"

python dgp_qp_example_001.py \
--image_path $IMAGE_PATH \
--class $CLASS \
--seed 0 \
--dgp_mode colorization \
--update_G \
--ftr_num 7 7 7 7 7 \
--ft_num 2 3 4 5 6 \
--lr_ratio 0.7 0.7 0.8 0.9 1.0 \
--w_D_loss 1 1 1 1 1 \
--w_nll 0.02 \
--w_mse 0 0 0 0 0 \
--select_num 2 \
--sample_std 0.5 \
--iterations 200 200 300 400 300 \
--G_lrs 5e-5 5e-5 5e-5 5e-5 2e-5 \
--z_lrs 2e-2 1e-2 5e-3 1e-3 1e-4 \
--use_in False False False False False \
--resolution 256 \
--weights_root pretrained \
--load_weights 256 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema \
--measurement_setting d
