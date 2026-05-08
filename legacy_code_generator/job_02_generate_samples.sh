#!/bin/sh -l
# FILENAME:  job_02_generate_samples

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=18
#SBATCH --time=2-23:30:00
#SBATCH --job-name generate_MIP_samples
#SBATCH --output=joboutput/job_02_generate_samples.out

module load anaconda
conda activate opt-ml-env

python 02_generate_samples.py setcover xCutoffBound_new -j 16  # number of available CPUs
echo "setcover Samples Generated"

# python 02_generate_samples.py cauctions -j 8
# echo "cauctions Samples Generated"

python 02_generate_samples.py facilities xCutoffBound_new -j 16
echo "facilities Samples Generated"

# python 02_generate_samples.py indset -j 8
# echo "indset Samples Generated"
