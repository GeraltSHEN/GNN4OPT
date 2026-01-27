#!/bin/bash -l
# FILENAME:  job_generate_args

#SBATCH -A canli
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-01:00:00
#SBATCH --job-name generate_args
#SBATCH --output=joboutput/job_generate_args.out

module load anaconda
conda activate opt-ml-env

DATASETS=(set_cover)

for DATASET in "${DATASETS[@]}"; do
  python default_args.py \
    --dataset "${DATASET}" \
    --cfg_idx 21 \
    --epochs 1 \
    --loss_option LambdaNDCGLoss2  \
    --relevance_type linear \
    --use_default_features true \
    --max_samples_per_split none
done

for DATASET in "${DATASETS[@]}"; do
  python default_args.py \
    --dataset "${DATASET}" \
    --cfg_idx 22 \
    --epochs 1 \
    --loss_option LambdaNDCGLoss2  \
    --relevance_type linear \
    --use_default_features false \
    --max_samples_per_split none
done
