#!/bin/bash
#SBATCH --job-name=realcause-tune-4
#SBATCH --mem=15000  # Requested Memory
#SBATCH --partition=gypsum-m40 # gypsum-m40      # Partition
#SBATCH -t 12:00:00  # Job time limit
#SBATCH --gres=gpu:m40:1
#SBATCH -o cluster/logs/realcause_tune/job-%j.out
#SBATCH -e cluster/logs/realcause_tune/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate realcause-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u train_generator.py $@
