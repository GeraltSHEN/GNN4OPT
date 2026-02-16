#!/bin/bash -l
# FILENAME:  job_resume_train_overfit

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --time=5-1:30:00
#SBATCH --job-name train_raw_resume
#SBATCH --output=joboutput/job_resume_train_overfit.out

module load anaconda
conda activate opt-ml-env

DATASETS=("set_cover")
CFG_IDS=(41)

for DATASET in "${DATASETS[@]}"; do
  echo "** ${DATASET} dataset **"
  for CFG in "${CFG_IDS[@]}"; do
    MODEL_SUFFIX="resume_from_${DATASET}_cfg${CFG}"
    echo "Training ${DATASET} cfg ${CFG}"
    python train.py --dataset "${DATASET}" --cfg_idx "${CFG}" --resume
    echo "Evaluating ${DATASET} cfg ${CFG}"
    python eval.py --dataset "${DATASET}" --cfg_idx "${CFG}" --model_suffix "${MODEL_SUFFIX}" --eval_split test
  done
done
