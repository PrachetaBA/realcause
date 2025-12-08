#!/bin/bash
#SBATCH --job-name=sbice-metrics  # Job name
#SBATCH --mem=4G  # Requested Memory
#SBATCH --partition=cpu      # Partition
#SBATCH -t 1-00:00:00  # Job time limit
#SBATCH -o ../logs/sbice_metrics/job-%j.out
#SBATCH -e ../logs/sbice_metrics/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate rc-ff-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u -m sbice.compute_metrics --experiment_identifier $1
