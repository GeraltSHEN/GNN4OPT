#!/bin/sh -l
# FILENAME:  job_augment_dataset

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=10
#SBATCH --time=5-1:30:00
#SBATCH --job-name job_augment_dataset
#SBATCH --output=joboutput/job_augment_dataset.out

module load anaconda
conda activate opt-ml-env

# python generate_top8_regression_targets.py --dual_options 1,2,3,4 --universal_cutoffbound 1e4 \
#       --quick_test \
#       --option_time_limit_sec 600

python generate_top8_regression_targets.py --dual_options 1,2 --universal_cutoffbound 1e4 --num_workers 8