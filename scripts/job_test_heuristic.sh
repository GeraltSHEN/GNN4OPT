#!/bin/sh -l
# FILENAME:  job_test_heuristic

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=0
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name test_heuristic
#SBATCH --output=joboutput/job_test_heuristic.out

module load anaconda
conda activate opt-ml-env

module load anaconda
conda activate opt-ml-env
python heuristics/test.py \
  --config cfg/set_cover_41111 \
  --use_trained_model \
  --split all \
  --max_samples 10 \
  --k 8 \
  --verbose_every 2 \
  --results_json heuristics/results/set_cover_41111_all.json
