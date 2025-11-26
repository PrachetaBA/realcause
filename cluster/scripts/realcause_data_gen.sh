#!/bin/bash
#SBATCH --job-name=rc-data-gen  # Job name
#SBATCH --mem=4G  # Requested Memory
#SBATCH --partition=cpu      # Partition
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ../logs/realcause_data_gen/job-%j.out
#SBATCH -e ../logs/realcause_data_gen/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate rc-ff-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u -m src.generate_datasets --config_file configs/realcause_experiments.yaml --experiment_identifier $1
