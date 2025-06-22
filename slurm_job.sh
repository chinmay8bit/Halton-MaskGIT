#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=resgpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cp524

export HF_HOME="/vol/bitbucket/cp524/hf_cache"

# for offline loading only
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Activate virtual environment
export PATH=/vol/bitbucket/cp524/dev/papers_with_code/Halton-MaskGIT/venv/bin:$PATH
source /vol/bitbucket/cp524/dev/papers_with_code/Halton-MaskGIT/venv/bin/activate

# Set up CUDA
source /vol/cuda/12.5.0/setup.sh

# Navigate to script directory
cd /vol/bitbucket/cp524/dev/papers_with_code/Halton-MaskGIT

export PYTHONUNBUFFERED=1

# Run training notebook
# python src/inference_with_SMC.py
python inference_remdm_using_SMC.py
