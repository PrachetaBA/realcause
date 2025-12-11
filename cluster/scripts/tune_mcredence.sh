#!/bin/bash
#SBATCH --job-name=mcredence-tune
#SBATCH --mem=8G # Requested Memory
#SBATCH --partition=cpu # Partition
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ../logs/mcredence_tune/job-%j.out
#SBATCH -e ../logs/mcredence_tune/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate rc-cred-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
# srun python -u -m modified_credence_tuning.modified_credence_tune --dataset_name lalonde --dataset_identifier psid1 --sample_size None --experiment_identifier $1 --rc_model_path results/GenModelCkpts/lalonde/psid1/save --outcome_model $2 --treatment_model $3
srun python -u -m modified_credence_tuning.modified_credence_tune --dataset_name postgres --dataset_identifier linear --sample_size 3000 --experiment_identifier $1 --rc_model_path results/realcause_models/postgres_linear_3000/default --outcome_model $2 --treatment_model $3
