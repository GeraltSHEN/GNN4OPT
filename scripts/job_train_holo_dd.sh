#!/bin/sh -l
# FILENAME:  job_train_holo_dd

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name train_holo_dd
#SBATCH --output=joboutput/job_train_holo_dd.out

module load anaconda
conda activate opt-ml-env

python disjunctive_dual/train.py --dataset set_cover --cfg_idx 0
python disjunctive_dual/eval.py --dataset set_cover --cfg_idx 0 --eval_split test
# python disjunctive_dual/train.py --dataset set_cover --cfg_idx 1
# python disjunctive_dual/eval.py --dataset set_cover --cfg_idx 1 --eval_split test

