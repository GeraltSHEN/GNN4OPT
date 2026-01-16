#!/bin/sh -l
# FILENAME:  job_train_raw2

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name train_raw2
#SBATCH --output=joboutput/job_train_raw2.out

module load anaconda
conda activate opt-ml-env

# DATASETS=("set_cover" "cauctions" "facilities" "indset")
# CFG_IDS=(10 11 12 13 14 15)
DATASETS=("cauctions" "facilities" "indset")
CFG_IDS=(10 11 12 13 14 15)

for DATASET in "${DATASETS[@]}"; do
  echo "** ${DATASET} dataset **"
  for CFG in "${CFG_IDS[@]}"; do
    echo "Training ${DATASET} cfg ${CFG}"
    python train.py --dataset "${DATASET}" --cfg_idx "${CFG}"
    echo "Evaluating ${DATASET} cfg ${CFG}"
    python eval.py --dataset "${DATASET}" --cfg_idx "${CFG}" --eval_split test
  done
done
