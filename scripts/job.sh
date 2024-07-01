#!/bin/bash
#SBATCH --job-name=CGCN
#SBATCH -p GPUMAT01
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
echo $(hostname) $CUDA_VISIBLE_DEVICES

PD=data/douban
edge_ratio=0.9

for anchor_ratio in 0.3 0.6 0.9
do
python train.py \
--anchor ${PD}/anchor${anchor_ratio}.pkl \
--split_path ${PD}/split${edge_ratio}.pkl \
--dim 128 \
--lr 0.001 \
--pre_epochs 0 \
--epochs 500 \
--margin 0.9 \
--alpha 0.1 \
--verbose 0 \
--edge_type pa1 \
--nodeattr adj \
--expand_edges True
done