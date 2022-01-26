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


void print_model(int *model, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%d ", model[i * N + j]);
    }
    printf("\n");
  }
}


/**
 * Instead of using if's or implementing a struct of any sort, we will be using this.
 * Rolls the array into itself. Index `N` points to `0` and index `-1` points to `N-1`
 * Ask for the indices and the size of the model.
 * `+N` takes care of negative indices,
 * `%N` takes care of indices greater than the size of the array.
 */
__device__ int get_model(int *model, int i, int j, int N) {
  int x = (i + N) % N;
  int y = (j + N) % N;

  return model[x * N + y];
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
__global__ void simulate_model(int *before, int *after, int N, int B) {
  int i = blockIdx.x / (N / B); /* the concurrent batch of rows on the 2D table the thread belongs to */
  int j = blockIdx.x % (N / B); /* the concurrent batch of columns on the 2D table the thread belongs to */

  // The northwest position of each block. Iterate through the rest of the moments.
  int index = (i * N + j) * B;

  int neighbours[4];
  for (int mx = 0; mx < B; mx++) {
    for (int my = 0; my < B; my++) {
      if (i + mx < N && j + my < N) {
        int block_index = index + mx * N + my;

        // Decompose the `index` parameter into row and column indices.
        // Add mx and my to them.
        neighbours[0] = get_model(before, i*B + mx, j*B + my + 1, N);
        neighbours[1] = get_model(before, i*B + mx, j*B + my - 1, N);
        neighbours[2] = get_model(before, i*B + mx + 1, j*B + my, N);
        neighbours[3] = get_model(before, i*B + mx - 1, j*B + my, N);

        after[block_index] = sign(before[block_index], neighbours, 4);
      }
    }
  }
}


int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: v2.out N B K, where:\
      \n -N is size,\
      \n -B is size of moment block and\
      \n -K is iterations\n"
    );
    return -1;
  }
  int N = atoi(argv[1]);
  int B = atoi(argv[2]);
  int K = atoi(argv[3]);
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
  
  for (int iter = 0; iter < K; iter++) {
    simulate_model<<<N * N / (B * B), 1>>>(d_before, d_after, N, B);
    // Pass the `after` values to the `before` model for the next iteration.
    cudaMemcpy(d_before, d_after, size, cudaMemcpyDeviceToDevice);

    cudaMemcpy(after, d_after, size, cudaMemcpyDeviceToHost);
    printf("\nMODEL AFTER ITERATION iter = %d\n", iter);
    print_model(after, N);
  }

  return 0;
}