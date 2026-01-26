#!/bin/sh -l
# FILENAME:  job_train_new2

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name train_new2
#SBATCH --output=joboutput/job_train_new2.out

module load anaconda
conda activate opt-ml-env

DATASETS=("cauctions" "facilities" "indset")
CFG_IDS=(20)

for DATASET in "${DATASETS[@]}"; do
  echo "** ${DATASET} dataset **"
  for CFG in "${CFG_IDS[@]}"; do
    echo "Training ${DATASET} cfg ${CFG}"
    python train.py --dataset "${DATASET}" --cfg_idx "${CFG}"
    echo "Evaluating ${DATASET} cfg ${CFG}"
    python eval.py --dataset "${DATASET}" --cfg_idx "${CFG}" --eval_split test
  done
done
