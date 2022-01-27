#!/bin/bash
#SBATCH --job-name=cuda_test
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00

module load gcc/7.3.0 cuda/10.0.130

nvcc v2_test.cu -o v2.out

nvidia-smi
for blocksize in 4 16 32
  do
    for i in {0..10}
    do
      ./v2.out 2048 $blocksize 1000
  done
done