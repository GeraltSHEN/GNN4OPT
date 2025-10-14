#!/bin/sh -l
# FILENAME:  job_train_holo

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name train_holo
#SBATCH --output=joboutput/job_train_holo.out

module load anaconda
conda activate opt-ml-env

python train.py --dataset set_cover --cfg_idx 2
python eval.py --dataset set_cover --cfg_idx 2 --eval_split all