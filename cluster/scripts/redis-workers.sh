#!/bin/bash
#SBATCH --job-name=redis-workers
#SBATCH -p cpu
#SBATCH --mem=1G
#SBATCH -t 2-00:00:00
#SBATCH -o ../logs/redis-samplers/worker-%j.out
#SBATCH -e ../logs/redis-samplers/worker-%j.err
#SBATCH --export=ALL,PYTHONPATH=/scratch3/workspace/pboddavarama_umass_edu-sbice/realcause

# prepare environment, e.g. set path
module load conda/latest
conda activate realcause-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
# run
abc-redis-worker --host="$1" --port="$2" --runtime=48h