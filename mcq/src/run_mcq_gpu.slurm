#!/bin/bash
#SBATCH --job-name=mcq_llama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=mcq_%j.out
#SBATCH --error=mcq_%j.err

module --force purge
module load release/23.04
module load GCCcore/11.2.0
module load Python/3.9.6
module load CUDA

cd /data/walrus/ws/ansh099g-llm_project/cultural-qa-benchmark/mcq
source venv/bin/activate

cd src
python run_mcq.py
