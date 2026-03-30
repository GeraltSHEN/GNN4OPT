#!/bin/sh -l
# FILENAME:  job_eval_aug_dual6

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name 6_eval_aug_dual
#SBATCH --output=joboutput/job_eval_aug_dual6.out

module load anaconda
conda activate opt-ml-env

DATASETS=("set_cover")
CFG_IDS=(68)

for DATASET in "${DATASETS[@]}"; do
  echo "** ${DATASET} dataset **"
  for CFG in "${CFG_IDS[@]}"; do
    echo "eval ${DATASET} cfg ${CFG}"
    python eval_heuristics.py --dataset "${DATASET}" --cfg_idx "${CFG}"
  done
done