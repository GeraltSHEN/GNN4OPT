#!/bin/sh -l
# FILENAME:  job_02_generate_samples

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-23:30:00
#SBATCH --job-name generate_MIP_samples
#SBATCH --output=joboutput/job_02_generate_samples.out

module load anaconda
conda activate opt-ml-env

python 02_generate_samples.py setcover -j 8  # number of available CPUs