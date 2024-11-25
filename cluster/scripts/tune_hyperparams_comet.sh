#!/bin/bash
#SBATCH --job-name=realcause-tune  # Job name
#SBATCH --mem=16000  # Requested Memory
#SBATCH --partition=cpu # Partition
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -o cluster/logs/realcause_tune/job-%j.out
#SBATCH -e cluster/logs/realcause_tune/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate /work/pi_jensen_umass_edu/pboddavarama_umass_edu/.conda/envs/realcause-exact

cd /work/pi_jensen_umass_edu/pboddavarama_umass_edu/nfl/realcause/
python -u train_generator_comet.py $@