#!/bin/sh -l
# FILENAME:  job_resume_train_holognn_try_1bsz

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name train_holognn_resume_1bsz
#SBATCH --output=joboutput/job_resume_train_holognn_1bsz.out

module load anaconda
conda activate opt-ml-env

CFG_IDX=100
DATASETS="set_cover"

for DATASET in ${DATASETS}; do
  MODEL_SUFFIX="resume_from_${DATASET}_cfg${CFG_IDX}"

  python train.py --dataset "${DATASET}" --cfg_idx "${CFG_IDX}" --resume \
    --save_every 100000 --eval_every 100000 --print_every 100000
  python eval.py --dataset "${DATASET}" --cfg_idx "${CFG_IDX}" --model_suffix "${MODEL_SUFFIX}" --eval_split test
done
