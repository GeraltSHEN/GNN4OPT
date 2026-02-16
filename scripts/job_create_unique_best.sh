#!/bin/sh -l
# FILENAME:  job_create_unique_best

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=5:30:00
#SBATCH --job-name create_unique_best
#SBATCH --output=joboutput/job_create_unique_best.out

module load anaconda
conda activate opt-ml-env

python create_unique_best_dataset.py