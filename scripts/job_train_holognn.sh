#!/bin/sh -l
# FILENAME:  job_train_holognn

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name train_holognn
#SBATCH --output=joboutput/job_train_holognn.out

module load anaconda
conda activate opt-ml-env

python train.py --dataset set_cover --cfg_idx 3
python eval.py --dataset set_cover --cfg_idx 3 --eval_split test

echo "** cauctions dataset **"
python train.py --dataset cauctions --cfg_idx 3
python eval.py --dataset cauctions --cfg_idx 3 --eval_split test

echo "** facilities dataset **"
python train.py --dataset facilities --cfg_idx 3
python eval.py --dataset facilities --cfg_idx 3 --eval_split test

echo "** indset dataset **"
python train.py --dataset indset --cfg_idx 3
python eval.py --dataset indset --cfg_idx 3 --eval_split test