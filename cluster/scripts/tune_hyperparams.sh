#!/bin/bash
#SBATCH --job-name=realcause-tune-4
#SBATCH --mem=15000  # Requested Memory
#SBATCH --partition=gypsum-m40 # gypsum-m40      # Partition
#SBATCH -t 12:00:00  # Job time limit
#SBATCH --gres=gpu:m40:1
#SBATCH -o cluster/logs/realcause_tune/job-%j.out
#SBATCH -e cluster/logs/realcause_tune/job-%j.err

# Load the necessary modules
module load miniconda/22.11.1-1
conda activate /work/pi_jensen_umass_edu/pboddavarama_umass_edu/.conda/envs/realcause-exact

cd /work/pi_jensen_umass_edu/pboddavarama_umass_edu/nfl/realcause/
python -u train_generator.py $@