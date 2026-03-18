#!/bin/sh -l
# FILENAME:  job_run_heuristic_policy

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name job_run_heuristic_policy
#SBATCH --output=joboutput/job_run_heuristic_policy.out

module load anaconda
conda activate opt-ml-env

DATASETS=("set_cover")
CFG_IDS=(62)

for DATASET in "${DATASETS[@]}"; do
  echo "** ${DATASET} dataset **"
  for CFG in "${CFG_IDS[@]}"; do
    echo "Training ${DATASET} cfg ${CFG}"
    python train_heuristics.py --dataset "${DATASET}" --cfg_idx "${CFG}"
  done
done