#!/bin/sh -l
# FILENAME:  job_train_holognn_try_1bsz

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name train_holognn_try_1bsz
#SBATCH --output=joboutput/job_train_holognn_try_1bsz.out

module load anaconda
conda activate opt-ml-env

python train.py --dataset set_cover --cfg_idx 99
python eval.py --dataset set_cover --cfg_idx 99 --eval_split test