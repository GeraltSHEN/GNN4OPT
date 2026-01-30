#!/bin/sh -l
# FILENAME:  job_check_bad

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name check_bad
#SBATCH --output=joboutput/job_check_bad.out

module load anaconda
conda activate opt-ml-env

python bad_samples/find_bad_samples.py