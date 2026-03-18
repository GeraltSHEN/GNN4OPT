#!/bin/sh -l
# FILENAME:  job_check_heuristics

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-40gb
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-1:30:00
#SBATCH --job-name check_heuristics
#SBATCH --output=joboutput/job_check_heuristics.out

module load anaconda
conda activate opt-ml-env

# python heuristics/test.py --config cfg/set_cover_54 \
#     --verbose_every 25 \
#     --use_trained_model

start=$(date +%s)
python new_heuristics/test.py --config cfg/set_cover_54 \
    --verbose_every 25 \
    --use_trained_model
end=$(date +%s)
echo "Elapsed: $((end - start)) seconds"