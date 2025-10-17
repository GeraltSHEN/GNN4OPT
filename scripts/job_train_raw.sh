#!/bin/sh -l
# FILENAME:  job_train_raw

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name train_raw
#SBATCH --output=joboutput/job_train_raw.out

module load anaconda
conda activate opt-ml-env

# python train.py --dataset set_cover --cfg_idx 1
# python eval.py --dataset set_cover --cfg_idx 1 --eval_split all

echo "** cauctions dataset **"
python train.py --dataset cauctions --cfg_idx 1
python eval.py --dataset cauctions --cfg_idx 1 --eval_split all

echo "** facilities dataset **"
python train.py --dataset facilities --cfg_idx 1
python eval.py --dataset facilities --cfg_idx 1 --eval_split all

echo "** indset dataset **"
python train.py --dataset indset --cfg_idx 1
python eval.py --dataset indset --cfg_idx 1 --eval_split all