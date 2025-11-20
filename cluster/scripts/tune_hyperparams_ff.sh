#!/bin/bash
#SBATCH --job-name=ff_htune
#SBATCH --nodes=10
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=../logs/ff_htune/job-%j.log
#SBATCH --error=../logs/ff_htune/job-%j.err

######### Set job-specific variables #########
dataset_name=$1 # 'e.g. n_acic_4'
dataset_identifier=$2 # 'e.g. linear'
sample_size=$3 # 'e.g., 3000 or all'
causal_model=$4 # 'e.g. location_translation'
#############################################

date;hostname;id;pwd
cd /scratch3/workspace/pboddavarama_umass_edu-sbice/frugal-flows/

module load conda/latest
conda activate /work/pi_jensen_umass_edu/pboddavarama_umass_edu/pba-conda/envs/frugalflows

echo 'activating virtual environment'
which python

train_file='hyperparameter_sweep.py'
echo 'train_file:' $train_file

project_name="frugal-flows-${dataset_name}-${dataset_identifier}-${sample_size}"
echo 'project_name:' $project_name

config_yaml="tuning_hyperparameters/${dataset_name}_${dataset_identifier}_${sample_size}_${causal_model}.yaml"
echo 'config:' $config_yaml

echo 'running script'
python wandb_slurm.py $config_yaml $train_file $project_name
