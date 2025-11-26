#!/bin/sh -l
# FILENAME:  job_speed_and_memory_test.sh

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --job-name speed_memory_test
#SBATCH --output=joboutput/job_speed_memory_test.out

module load anaconda
conda activate opt-ml-env

python disjunctive_dual/speed_and_memory_test.py
