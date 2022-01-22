#!/bin/bash
#SBATCH --job-name=cuda_test
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10

module load gcc/7.3.0 cuda/10.0.130

nvcc v1_test.cu -o v1

nvidia-smi
./v1 64 16 2

