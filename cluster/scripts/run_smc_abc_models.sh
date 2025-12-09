#!/bin/bash
#SBATCH --job-name=smc_abc_models  # Job name
#SBATCH --mem=16G  # Requested Memory
#SBATCH --partition=cpu      # Partition
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ../logs/smc_abc_models_runs/job-%j.out
#SBATCH -e ../logs/smc_abc_models_runs/job-%j.err

module load conda/latest
conda activate rc-ff-sbi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
# export JAX_PLATFORMS=cpu

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u -m sbi.smc_abc_models --config $1 --expt_num $2 --sampler redis --redis_server $3 --redis_port $4
