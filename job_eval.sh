#!/bin/sh -l
# FILENAME:  job_eval

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --job-name try_eval_code
#SBATCH --output=joboutput/job_eval.out

module load anaconda
conda activate opt-ml-env

python eval.py --dataset set_cover --cfg_idx 0 --eval_split all