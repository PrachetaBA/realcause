#!/bin/bash
#SBATCH --job-name=smc_abc_models_gpu  # Job name
#SBATCH --mem=48G  # Requested Memory
#SBATCH --partition=gpu      # Partition
#SBATCH --gres=gpu:1
#SBATCH --constraint=sm_61  # Required for JAX compatibility
#SBATCH -t 2-00:00:00  # Job time limit
#SBATCH -o ../logs/smc_abc_models_runs/job-%j.out
#SBATCH -e ../logs/smc_abc_models_runs/job-%j.err

module load conda/latest
conda activate rc-ff-sbi

# Set JAX/XLA environment variables for GPU memory management
# These MUST be set before JAX is imported
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3  # Reduced from 0.7 to leave more headroom
export XLA_FLAGS=--xla_gpu_graph_level=0
# Note: Slurm automatically sets CUDA_VISIBLE_DEVICES when using --gres=gpu:1

# Optional: Clear GPU memory from previous processes (if nvidia-smi is available)
# This helps prevent memory fragmentation issues
if command -v nvidia-smi &> /dev/null; then
    echo "GPU status before job:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
fi

# Print GPU info for debugging
echo "=== GPU Information ==="
nvidia-smi

echo "======================="
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
echo "======================="

# Debug: print XLA env vars to verify they're set
echo "=== XLA Environment Variables ==="
echo "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
echo "XLA_PYTHON_CLIENT_ALLOCATOR=$XLA_PYTHON_CLIENT_ALLOCATOR"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "=================================="

cd /scratch3/workspace/pboddavarama_umass_edu-sbice/realcause/
python -u -m sbi.smc_abc_models --config $1 --expt_num $2 --sampler redis --redis_server $3 --redis_port $4
