#!/bin/bash
#SBATCH --job-name=estimators-causal  # Job name
#SBATCH --mem=16G  # Requested Memory
#SBATCH --partition=cpu      # Partition
#SBATCH -t 1-00:00:00  # Job time limit
#SBATCH -o ../logs/causal_estimators/job-%j.out
#SBATCH -e ../logs/causal_estimators/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate /work/pi_jensen_umass_edu/pboddavarama_umass_edu/pba-conda/envs/rpy

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
type_of_run=$1
if [ "$type_of_run" == "source" ]; then
    python -u -m sbice.get_ate_estimates_models --experiment_config $2 --experiment_number $3 --observed_data --set_of_estimators $4 --smc_expt_id $5
elif [ "$type_of_run" == "generated" ]; then
    python -u -m sbice.get_ate_estimates_models --experiment_config $2 --experiment_number $3  --posterior_or_prior $4 --set_of_estimators $5 --smc_expt_id $6
else
    echo "Invalid type of run"
    exit 1
fi
