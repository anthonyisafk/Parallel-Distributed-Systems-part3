#!/bin/bash
#SBATCH --job-name=cuda_test
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00

module load gcc/7.3.0 cuda/10.0.130

nvcc v2_test.cu -o v2.out

nvidia-smi
for size in 64 256 512 1024 2048
do
  for blocksize in 4 16 32
  do
    for i in {0..10}
    do
      ./v2.out $size $blocksize 1000
    done
  done
done
