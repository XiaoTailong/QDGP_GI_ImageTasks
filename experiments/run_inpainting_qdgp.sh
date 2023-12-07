#!/bin/bash
#SBATCH --job-name=DGP_qp
#SBATCH --partition=CLUSTER
#SBATCH -t 7-00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodelist=compute-0-2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -o ./output/%x-%j.out
#SBATCH -e ./output/%x-%j.err
module load python/anaconda3
module load cuda/cuda-11.4
source activate tf_tc


#IMAGE_PATH="/data/ILSVRC2012_val_00022130.JPEG"
#CLASS="511"


# cat2 class 282
# bee2 class 309

#IMAGE_PATH="/data/ILSVRC2012_val_00001970.JPEG"
#CLASS="737"

#IMAGE_PATH="/data/cat1.JPEG"
#CLASS="-1"


#IMAGE_PATH="/data/388_pandas_8.jpeg"
#CLASS="388"
#IMAGE_PATH="/data/388_pandas_9.jpeg"
#CLASS="388"


IMAGE_PATH="/data/216_dog_14.jpeg"
CLASS="216"

#IMAGE_PATH="/data/355_anmial_13.jpeg"
#CLASS="-1"
#IMAGE_PATH="/data/157_dog_16.jpeg"
#CLASS="157"


python dgp_qp_example_001.py \
--image_path $IMAGE_PATH \
--class $CLASS \
--seed 42 \
--dgp_mode inpainting \
--update_G \
--update_embed \
--ftr_num 8 8 8 8 8 \
--ft_num 7 7 7 7 7  \
--lr_ratio 1.0 1.0 1.0 1.0 1.0  \
--w_D_loss 1 1 1 1 0.5 \
--w_nll 0.02 \
--w_mse 1 1 1 1 10 \
--select_num 1 \
--sample_std 0.3 \
--iterations 200 200 200 200 200 \
--G_lrs 5e-5 5e-5 2e-5 2e-5 1e-5 \
--z_lrs 5e-2 5e-3 1e-3 5e-4 1e-4  \
--use_in False False False False False \
--resolution 256 \
--weights_root pretrained \
--load_weights 256 \
--G_ch 96 --D_ch 96 \
--G_shared \
--hier --dim_z 120 --shared_dim 128 \
--skip_init --use_ema \
--measurement_setting d
