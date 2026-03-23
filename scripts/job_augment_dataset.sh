#!/bin/sh -l
# FILENAME:  job_augment_dataset

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name job_augment_dataset
#SBATCH --output=joboutput/job_augment_dataset.out

module load anaconda
conda activate opt-ml-env

python generate_top8_regression_targets.py