#!/bin/bash
#SBATCH --job-name=ff_tune
#SBATCH --nodes=2  # 10 for full sweep; 1 for testing
#SBATCH --partition=cpu  # cpu
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
conda activate rc-ff-sbi

export JAX_PLATFORMS=cpu

echo 'activating virtual environment'
which python

train_file='frugal_flows_tuning/hyperparameter_sweep.py'
echo 'train_file:' $train_file

project_name="frugalflows-${dataset_name}-${dataset_identifier}-${sample_size}"
echo 'project_name:' $project_name

config_yaml="ff_hyperparameter_tuning/${dataset_name}_${dataset_identifier}_${sample_size}_${causal_model}.yaml"
echo 'config:' $config_yaml

echo 'running script'
python -u frugal_flows_tuning/wandb_slurm.py $config_yaml $train_file $project_name
