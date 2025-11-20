#!/bin/bash
#SBATCH --job-name=smc_abc  # Job name
#SBATCH --mem=2G  # Requested Memory
#SBATCH --partition=cpu      # Partition
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ../logs/smc_abc_runs/job-%j.out
#SBATCH -e ../logs/smc_abc_runs/job-%j.err

module load conda/latest
conda activate realcause-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u -m sbi.smc_abc_rc --config configs/experiments_lalonde_psid.yaml --expt_num $1 --sampler redis --redis_server $2 --redis_port $3