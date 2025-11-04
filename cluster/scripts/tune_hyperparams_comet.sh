#!/bin/bash
#SBATCH --job-name=realcause-tune  # Job name
#SBATCH --mem=16G  # Requested Memory
#SBATCH --partition=cpu # Partition
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o cluster/logs/realcause_tune/job-%j.out
#SBATCH -e cluster/logs/realcause_tune/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate realcause-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u train_generator_comet.py $@