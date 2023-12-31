#!/bin/bash
#SBATCH --job-name=QDGP
#SBATCH --partition=CLUSTER
#SBATCH -t 7-00:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodelist=compute-0-2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -o ./output/%x-%j.out
#SBATCH -e ./output/%x-%j.err

### export MASTER_PORT=12342
### export WORLD_SIZE=1
### echo "NODELIST="${SLURM_NODELIST}
### master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
### export MASTER_ADDR=$master_addr
### echo "MASTER_ADDR="$MASTER_ADDR

module load python/anaconda3
module load cuda/cuda-11.4
source activate tf_tc

D='64 128 256 512 1024 2048'
M='t'   ### 这个o好像还是,还是z与y加一起好很多？
OBJ='wing butterfly epithelia'

for obj in $OBJ
do
  for m in $M
  do
    for d in $D
    do
      python QDGP_untrain_no_quantum_train.py \
      --class -1 \
      --seed 1314 \
      --update_G \
      --update_embed \
      --lr_ratio 1.0 \
      --iterations 1000 1000 1000 1000 1000 \
      --G_lrs 1.4e-5 1.4e-5 1.4e-5 0.7e-5 0.35e-5 \
      --z_lrs 7e-3 7e-4 7e-4 7e-5 7e-6  \
      --use_in False False False False False \
      --resolution 256 \
      --weights_root pretrained \
      --load_weights 256 \
      --G_ch 96 \
      --G_shared \
      --hier --dim_z 120 --shared_dim 128 \
      --skip_init --use_ema \
      --n_qubits 8 \
      --n_qlayers 3 \
      --dims $d \
      --n_heads 3 \
      --measurement_setting $m \
      --object $obj
      done
  done
done
