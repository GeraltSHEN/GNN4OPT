#!/bin/sh -l
# FILENAME:  job_verify_dataset

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name verify_dataset
#SBATCH --output=joboutput/job_verify_dataset.out

module load anaconda
conda activate opt-ml-env

python disjunctive_dual/verify_dataset.py
# python check_data_dist.py --root data/data_dist/set_cover
