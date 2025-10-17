#!/bin/sh -l
# FILENAME:  job_resume_train_raw

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name train_raw_resume
#SBATCH --output=joboutput/job_resume_train_raw.out

module load anaconda
conda activate opt-ml-env

MODEL_SUFFIX=resume_from_set_cover_cfg1

python train.py --dataset set_cover --cfg_idx 1 --resume
python eval.py --dataset set_cover --cfg_idx 1 --model_suffix "${MODEL_SUFFIX}" --eval_split test
