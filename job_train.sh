#!/bin/sh -l
# FILENAME:  job_train

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name try_train_code
#SBATCH --output=joboutput/job_train.out

module load anaconda
conda activate opt-ml-env

python train.py --dataset set_cover --cfg_idx 0 --eval_every 1000 --print_every 1000 --save_every 10000
python eval.py --dataset set_cover --cfg_idx 0 --eval_split test