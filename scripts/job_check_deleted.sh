#!/bin/sh -l
# FILENAME:  job_check_deleted

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=240G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name check_deleted
#SBATCH --output=joboutput/job_check_deleteds.out

module load anaconda
conda activate opt-ml-env

DATASETS=("set_cover")
CFG_IDS=(60)

for DATASET in "${DATASETS[@]}"; do
  echo "** ${DATASET} dataset **"
  for CFG in "${CFG_IDS[@]}"; do
    echo "Training ${DATASET} cfg ${CFG}"
    python train.py --dataset "${DATASET}" --cfg_idx "${CFG}"
    echo "Evaluating ${DATASET} cfg ${CFG}"
    python eval.py --dataset "${DATASET}" --cfg_idx "${CFG}" --eval_split test
  done
done