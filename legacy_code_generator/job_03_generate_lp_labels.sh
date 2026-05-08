#!/bin/sh -l
# FILENAME:  job_03_generate_lp_labels

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=10
#SBATCH --time=5-1:30:00
#SBATCH --job-name 03_generate_lp_labels
#SBATCH --output=joboutput/job_03_generate_lp_labels.out

module load anaconda
conda activate opt-ml-env

# Option 1 is now embedded by 02_generate_samples.py (patched PySCIPOpt).
# Keep this step only for option 2
python 03_generate_lp_labels.py --data_path data/samples/facilities_xCutoffBound/100_100_5 --dual_options 2 --num_workers 8
