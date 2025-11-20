#!/bin/bash
#SBATCH --job-name=ff_tune
#SBATCH --partition=gpu  # cpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=../logs/frugalflows_tune/job-%j.log
#SBATCH --error=../logs/frugalflows_tune/job-%j.err

######### Set job-specific variables #########
dataset_name=$1 # 'e.g. n_acic_4'
dataset_identifier=$2 # 'e.g. linear'
sample_size=$3 # 'e.g., 3000 or all'
causal_model=$4 # 'e.g. location_translation'
#############################################

date;hostname;id;pwd
cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
module load conda/latest
module load cuda/12.1

# Set JAX environment variables to handle GPU compatibility
# If GPU kernels are incompatible, JAX will fallback to CPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# Uncomment the line below to force CPU usage if GPU issues persist
# export JAX_PLATFORMS=cpu

conda activate rc-ff-sbi

echo 'activating virtual environment'
which python

train_file='frugal_flows_tuning/hyperparameter_sweep.py'
echo 'train_file:' $train_file

project_name="frugalflows-${dataset_name}-${dataset_identifier}-${sample_size}"
echo 'project_name:' $project_name

config_yaml="ff_hyperparameter_tuning/${dataset_name}_${dataset_identifier}_${sample_size}_${causal_model}.yaml"
echo 'config:' $config_yaml

echo 'running script'
python -u frugal_flows_tuning/wandb_slurm_gpu.py $config_yaml $train_file $project_name
