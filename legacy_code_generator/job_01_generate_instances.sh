#!/bin/sh -l
# FILENAME:  job_01_generate_instances

#SBATCH -A canli
#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --partition=a100-80gb
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-23:30:00
#SBATCH --job-name generate_MIP_instances
#SBATCH --output=joboutput/job_01_generate_instances.out

module load anaconda
conda activate opt-ml-env

# python 01_generate_instances.py setcover
# echo "***** setcover Done *****"
python 01_generate_instances.py cauctions
echo "***** cauctions Done *****"
python 01_generate_instances.py facilities
echo "***** facilities Done *****"
python 01_generate_instances.py indset
echo "***** indset Done *****"