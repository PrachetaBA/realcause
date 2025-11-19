#!/bin/bash
#SBATCH --job-name=redis-server-launch
#SBATCH -p cpu
#SBATCH -t 2-00:00:00
#SBATCH -o ../logs/redis-samplers/server-%j.out
#SBATCH -e ../logs/redis-samplers/server-%j.err
#SBATCH --export=ALL,PYTHONPATH=/scratch3/workspace/pboddavarama_umass_edu-sbice/realcause

# prepare environment, e.g. set path
module load conda/latest
conda activate realcause-sbi

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
echo "Assigned host: $(hostname)"

echo "Starting Redis Server"
redis-server --bind 0.0.0.0 --port $1