#!/bin/bash
#SBATCH --job-name=cuda_test
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00

module load gcc/7.3.0 cuda/10.0.130

nvcc v1_test.cu -o v1.out

nvidia-smi
for i in {0..10}
do
  for size in 16 64 256 512 1024 2048
  do
  ./v1.out $size 1000
  done
done