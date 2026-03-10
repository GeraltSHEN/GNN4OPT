#!/bin/sh -l
# FILENAME:  job_test_heuristic_random

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name h_random
#SBATCH --output=joboutput/job_test_heuristic_max_random.out

module load anaconda
conda activate opt-ml-env

module load anaconda
conda activate opt-ml-env
python heuristics/test.py \
  --config cfg/set_cover_41 \
  --split test \
  --max_samples 1000 \
  --k 8 \
  --verbose_every 10 \
  --results_json heuristics/results/set_cover_41_test_max_random.json