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

DATASETS=(set_cover cauctions facilities indset)
# CFG_IDS=(10 11 12 13 14 15)

# for DATASET in "${DATASETS[@]}"; do
#   for CFG in "${CFG_IDS[@]}"; do
#     case "${CFG}" in
#       10) LOSS="PairwiseLogisticLoss" ;;
#       11) LOSS="NormalizedPairwiseLogisticLoss" ;;
#       12) LOSS="LambdaARPLoss2" ;;
#       13) LOSS="TierNormalizedLambdaARP2" ;;
#       14) LOSS="LambdaARPLoss2" ;;
#       15) LOSS="TierNormalizedLambdaARP2" ;;
#       *)
#         echo "Unsupported cfg_idx ${CFG}" >&2
#         exit 1
#         ;;
#     esac

#     if [ "${CFG}" -ge 10 ] && [ "${CFG}" -le 13 ]; then
#       REL_TYPE="linear"
#     else
#       REL_TYPE="exponential"
#     fi

#     echo "Generating defaults for ${DATASET} cfg ${CFG} with ${LOSS} (relevance_type=${REL_TYPE})"
#     python default_args.py \
#       --dataset "${DATASET}" \
#       --cfg_idx "${CFG}" \
#       --epochs 2 \
#       --loss_option "${LOSS}" \
#       --relevance_type "${REL_TYPE}" \
#       --max_samples_per_split none
#   done
# done

for DATASET in "${DATASETS[@]}"; do
  python default_args.py \
    --dataset "${DATASET}" \
    --cfg_idx 20 \
    --epochs 2 \
    --loss_option TierAwarePairwiseLogisticLoss  \
    --relevance_type linear \
    --max_samples_per_split none
done
