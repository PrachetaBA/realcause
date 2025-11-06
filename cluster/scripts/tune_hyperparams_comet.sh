#!/bin/bash
#SBATCH --job-name=realcause-tune  # Job name
#SBATCH --mem=32G  # Requested Memory
#SBATCH --partition=cpu # Partition
#SBATCH -t 1-00:00:00  # Job time limit
#SBATCH -o cluster/logs/realcause_tune/job-%A_%a.out
#SBATCH -e cluster/logs/realcause_tune/job-%A_%a.err
#SBATCH --array=0-4  # Run 5 parallel jobs (indices 0-4)

# Load the necessary modules
module load conda/latest
conda activate realcause-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/

# Create unique saveroot for each parallel run
SAVEROOT="tuned_models/${1}_${2}_${3:all}_run${SLURM_ARRAY_TASK_ID}"

# Run the tuning script - each array task runs independently
python -u train_generator_comet.py \
    --data "$1" \
    --data_identifier "${2:-}" \
    --sample_size "${3:all}" \
    --saveroot "$SAVEROOT" 