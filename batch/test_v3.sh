#!/bin/bash
#SBATCH --job-name=cuda_v3
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00

module load gcc/7.3.0 cuda/10.0.130

nvcc v3_test.cu -o v3.out

nvidia-smi
for size in 64 256 512 1024
do
  for blocksize in 4 16 32
  do
    for i in {0..10}
    do
      ./v3.out $size 1 $blocksize 1000
    done
  done
done