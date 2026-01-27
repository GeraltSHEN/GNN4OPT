#!/bin/sh -l
# FILENAME:  job_check_output

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name check_output
#SBATCH --output=joboutput/job_check_output.out

module load anaconda
conda activate opt-ml-env

DATASETS=("set_cover")

python check_output.py 