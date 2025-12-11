#!/bin/bash
#SBATCH --job-name=credence-tune
#SBATCH --mem=16G # Requested Memory
#SBATCH --partition=cpu # Partition
#SBATCH --cpus-per-task=4 # 4 CPUs per worker (matching scaling_config resources_per_worker)
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ../logs/credence_tune/job-%j.out
#SBATCH -e ../logs/credence_tune/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate rc-cred-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
# srun python -u -m credence_tuning.credence_tune --dataset_name lalonde --dataset_identifier psid1 --sample_size None --experiment_identifier $1 --rc_model_path results/GenModelCkpts/lalonde/psid1/save --outcome_model $2 --covariates_model $3
srun python -u -m credence_tuning.credence_tune --dataset_name postgres --dataset_identifier linear --sample_size 3000 --experiment_identifier $1 --rc_model_path results/realcause_models/postgres_linear_3000/default --outcome_model $2 --covariates_model $3
