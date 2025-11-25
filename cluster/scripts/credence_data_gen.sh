#!/bin/bash
#SBATCH --job-name=credence-data-gen
#SBATCH --mem=8G # Requested Memory
#SBATCH --partition=cpu # Partition
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ../logs/credence_data_gen/job-%j.out
#SBATCH -e ../logs/credence_data_gen/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate rc-cred-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u -m credence_tuning.credence_models_data_gen --gen_model credence --experiment_identifier $1
