#!/bin/bash
#SBATCH --job-name=ff-data-gen  # Job name
#SBATCH --mem=16G  # Requested Memory
#SBATCH --partition=gpu      # Partition
#SBATCH --gres=gpu:1
#SBATCH --constraint=sm_61  # Required for JAX compatibility
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ../logs/frugalflows_data_gen/job-%j.out
#SBATCH -e ../logs/frugalflows_data_gen/job-%j.err

# Load the necessary modules
module load conda/latest
conda activate rc-ff-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u -m frugal_flows_tuning.frugal_flows_data_gen --config_file configs/frugalflows_experiments.yaml --experiment_identifier $1
