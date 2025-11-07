#!/bin/bash
#=========================================================================================
# Slurm SBATCH Directives
#=========================================================================================
#SBATCH --job-name=arc3-inf-test
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64          # 8卡×每卡6核；视节点CPU调整为32/40/48/64
# #SBATCH --nodelist=sines-2-embody-ai-node-4,sines-2-embody-ai-node-1,sines-2-embody-ai-node-2,sines-2-embody-ai-node-3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                     # request for 1 GPU
#SBATCH --gpus=1
#SBATCH --time=72:00:00                  # max running time DD-HH:MM:SS
#SBATCH --output=/shared_work/arc3/slurm-logs/%x-%j.out
#SBATCH --error=/shared_work/arc3/slurm-logs/%x-%j.out
#SBATCH --mem=128G
#SBATCH --qos=high
# #SBATCH --exclusive
# #SBATCH --dependency=afterok:1048
#=========================================================================================

set -a
source /shared_work/arc3/ARC-AGI-3-Agents/.env
set +a

# set to shared huggingface cache in our cluster
export HF_HOME="/shared_work/hf"
# set set all user level cache to shared directory
mkdir -p "/shared_work/arc3/.cache"
export XDG_CACHE_HOME="/shared_work/arc3/.cache"

# 统一线程数
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# (Optional) clean module state and load toolchains you need
# module purge || true
# module load cuda/12.1  # uncomment if your code needs a CUDA module

# ---- Activate Conda environment ----
# Enable 'conda' command
module load conda
# Activate the env we just created
conda activate /shared_work/conda-pkgs/qianyi/arc3
# ---- Your workload ----
nvidia-smi || true
python -V

CMD=(vllm serve openai/gpt-oss-20b)

echo "Running: ${CMD[*]}"
srun --exclusive "${CMD[@]}"

echo "Job finished at $(date)"
