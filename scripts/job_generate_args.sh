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

DATASETS=("set_cover" "cauctions" "facilities" "indset")
MAX_SAMPLES_PER_SPLIT="none"

for DATASET in "${DATASETS[@]}"; do
  python default_args.py \
    --dataset "${DATASET}" \
    --cfg_idx 30 \
    --epochs 2 \
    --loss_option LambdaNDCGLoss1  \
    --relevance_type linear \
    --use_default_features true \
    --remove_bad_candidates false \
    --max_samples_per_split $MAX_SAMPLES_PER_SPLIT
done

for DATASET in "${DATASETS[@]}"; do
  python default_args.py \
    --dataset "${DATASET}" \
    --cfg_idx 31 \
    --epochs 2 \
    --loss_option LambdaNDCGLoss1  \
    --relevance_type linear \
    --use_default_features true \
    --remove_bad_candidates true \
    --max_samples_per_split $MAX_SAMPLES_PER_SPLIT
done

for DATASET in "${DATASETS[@]}"; do
  python default_args.py \
    --dataset "${DATASET}" \
    --cfg_idx 32 \
    --epochs 2 \
    --loss_option LambdaNDCGLoss1  \
    --relevance_type linear \
    --use_default_features false \
    --remove_bad_candidates false \
    --max_samples_per_split $MAX_SAMPLES_PER_SPLIT
done

for DATASET in "${DATASETS[@]}"; do
  python default_args.py \
    --dataset "${DATASET}" \
    --cfg_idx 33 \
    --epochs 2 \
    --loss_option LambdaNDCGLoss1  \
    --relevance_type linear \
    --use_default_features false \
    --remove_bad_candidates true \
    --max_samples_per_split $MAX_SAMPLES_PER_SPLIT
done



# for DATASET in "${DATASETS[@]}"; do
#   python default_args.py \
#     --dataset "${DATASET}" \
#     --cfg_idx 32 \
#     --epochs 2 \
#     --loss_option TierAwarePairwiseLogisticLoss  \
#     --relevance_type linear \
#     --use_default_features true \
#     --remove_bad_candidates false \
#     --max_samples_per_split $MAX_SAMPLES_PER_SPLIT
# done
