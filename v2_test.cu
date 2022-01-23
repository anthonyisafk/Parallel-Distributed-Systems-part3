/**
 * @file: v2_test.cu
 * ****************************************
 * @author: Antonios Antoniou
 * @email: aantonii@ece.auth.gr
 * ****************************************
 * @description: Simulate the Ising model for a system of size `n x n` and `k` iterations.
 * Each GPU thread simulates an iteration for a block of points.
 * ****************************************
 * Parallel and Distributed Systems - Electrical and Computer Engineering
 * 2022 Aristotle University Thessaloniki.
 */
#include <stdio.h>
#include <stdlib.h>


/**
 * Instead of using if's or implementing a struct of any sort, we will be using this.
 * Rolls the array into itself. Index `size` points to `0` and index `-1` points to `n-1`
 * Ask for the indices and the size of the model.
 * `+size` takes care of negative indices,
 * `%size` takes care of indices greater than the size of the array.
 */
__device__ int get_model(int *model, int i, int j, int size) {
  int x = (i + size) % size;
  int y = (j + size) % size;

  return model[x * size + y];
}


// Implement the sign function with a node's neighbours.
__device__ int sign(int self, int *neighbours, int neighbours_n) {
  int sum = self;
  for (int i = 0; i < neighbours_n; i++) {
    sum += neighbours[i];
  }

  if (sum > 0) {
    return 1;
  }
  else if (sum < 0) {
    return -1;
  } else {
    return 0; /* never happens, added for mathematic completeness */
  }
}


// Simulates the behavior of a block of points for a single iteration.
// @param b: specifies the block size.
__global__ void simulate_model(int *before, int *after, int size, int b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int neighbours[4];

  // The northwest position of each block. Iterate through the rest of the moments.
  int index = (i * size + j) * b;

  for (int mx = 0; mx < b; mx++) {
    for (int my = 0; my < b; my++) {
      if (i + mx < size && j + my < size) {
        int block_index = index + mx * size + my;

        // Decompose the `index` parameter into row and column indices.
        // Add mx and my to them.
        neighbours[0] = get_model(before, i*b + mx, j*b + my + 1, size);
        neighbours[1] = get_model(before, i*b + mx, j*b + my - 1, size);
        neighbours[2] = get_model(before, i*b + mx + 1, j*b + my, size);
        neighbours[3] = get_model(before, i*b + mx - 1, j*b + my, size);

        after[block_index] = sign(before[block_index], neighbours, 4);
      }
    }
  }
}


void print_model(int *model, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      printf("%d ", model[i * size + j]);
    }
    printf("\n");
  }
}


int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: v2.out N BS B K, where\n\
      \t>> N is size\n\
      \t>> BS is block size\n\
      \t>> B is size of moment block\n\
      \t>> K is number of iterations");
    return -1;
  }
  const int N = atoi(argv[1]);
  const int BLOCKSIZE = atoi(argv[2]);
  const int b = atoi(argv[3]);
  const int K = atoi(argv[4]);
  const int size = N * N * sizeof(int);

  int *model = (int *) malloc(size);
  int *after = (int *) malloc(size);
  for (int i = 0; i < N * N; i++) {
    model[i] = (rand() > RAND_MAX / 2) ? -1 : 1;
  }
  printf("MODEL BEFORE FIRST ITERATION\n");
  print_model(model, N);

  // The model before and after each iteration on the GPU.
  int *d_before, *d_after;
  cudaMalloc((void **)&d_before, size);
  cudaMalloc((void **)&d_after, size);
  cudaMemcpy(d_before, model, size, cudaMemcpyHostToDevice);
  
  dim3 dim_block(BLOCKSIZE, BLOCKSIZE);
  dim3 dim_grid(N/(dim_block.x * b), N/(dim_block.y * b));

  for (int iter = 0; iter < K; iter++) {
    simulate_model<<<dim_grid, dim_block>>>(d_before, d_after, N, b);
    // Pass the `after` values to the `before` model for the next iteration.
    cudaMemcpy(d_before, d_after, size, cudaMemcpyDeviceToDevice);

    cudaMemcpy(after, d_after, size, cudaMemcpyDeviceToHost);
    printf("\nMODEL AFTER ITERATION iter = %d\n", iter);
    print_model(after, N);
  }

  return 0;
}