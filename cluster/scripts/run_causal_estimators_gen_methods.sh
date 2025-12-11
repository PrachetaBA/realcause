#!/bin/bash
#SBATCH --job-name=estimators-causal  # Job name
#SBATCH --mem=16G  # Requested Memory
#SBATCH --partition=cpu      # Partition
#SBATCH -t 1-00:00:00  # Job time limit
#SBATCH -o ../logs/causal_estimators_gen_methods/job-%j.out
#SBATCH -e ../logs/causal_estimators_gen_methods/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate /work/pi_jensen_umass_edu/pboddavarama_umass_edu/pba-conda/envs/rpy

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u -m sbice.gen_methods_ate_estimates --dataset_name $1 --dataset_identifier $2 --sample_size $3 --experiment_identifier $4 --gen_method $5
