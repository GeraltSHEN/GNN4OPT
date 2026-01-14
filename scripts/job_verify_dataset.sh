#!/bin/sh -l
# FILENAME:  job_verify_dataset

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name verify_dataset
#SBATCH --output=joboutput/job_verify_dataset.out

module load anaconda
conda activate opt-ml-env

PARENT_DIR="data/data_dist"
DATASETS=("set_cover" "cauctions" "facilities" "indset")
CFG_IDS=(0 2)

for DATASET in "${DATASETS[@]}"; do
  for CFG in "${CFG_IDS[@]}"; do
    echo "Running verify_dataset for ${DATASET} cfg ${CFG}"
    python verify_dataset.py \
      --dataset "${DATASET}" \
      --cfg_idx "${CFG}" \
      --parent_test_stats_dir "${PARENT_DIR}/${DATASET}" \
      # --max_samples_per_split 100
  done

  echo "Aggregating distributions for ${DATASET}"
  python check_data_dist.py \
    --dataset "${DATASET}" \
    --cfg_idx "${CFG_IDS[@]}" \
    --parent_dir "${PARENT_DIR}"
done
