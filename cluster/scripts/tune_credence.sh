#!/bin/bash
#SBATCH --job-name=credence-tune
#SBATCH --mem=8G # Requested Memory
#SBATCH --partition=cpu # Partition
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ../logs/credence_tune/job-%j.out
#SBATCH -e ../logs/credence_tune/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate rc-cred-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
srun python -u -m credence_tuning.credence_tune --dataset_name lalonde --dataset_identifier psid1 --sample_size None --experiment_identifier $1 --rc_model_path results/GenModelCkpts/lalonde/psid1/save --outcome_model $2 --covariates_model $3
