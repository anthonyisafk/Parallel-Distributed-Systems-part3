/**
 * @file: v1_test.cu
 * ****************************************
 * @author: Antonios Antoniou
 * @email: aantonii@ece.auth.gr
 * ****************************************
 * @description: Simulate the Ising model for a system of size `n x n` and `k` iterations.
 * Each individual point for a single iteration is simulated in a GPU thread.
 * ****************************************
 * Parallel and Distributed Systems - Electrical and Computer Engineering
 * 2022 Aristotle University Thessaloniki.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


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


// Simulates the behavior of a single point for a single iteration.
__global__ void simulate_model(int *before, int *after, int N) {
  int index = blockIdx.x;
  int i = index / N; /* the concurrent batch of rows on the 2D table the thread belongs to */
  int j = index % N; /* the concurrent batch of columns on the 2D table the thread belongs to */
  int neighbours[4];

  if (i < N && j < N) {
    neighbours[0] = get_model(before, i, j + 1, N);
    neighbours[1] = get_model(before, i, j - 1, N);
    neighbours[2] = get_model(before, i + 1, j, N);
    neighbours[3] = get_model(before, i - 1, j, N);

    after[index] = sign(before[index], neighbours, 4);
  }
}


int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: v1.out N K, where N is size, and K is iterations\n");
    return -1;
  }
  int N = atoi(argv[1]);
  int K = atoi(argv[2]);
  const int size = N * N * sizeof(int);

  int *model = (int *) malloc(size);
  int *after = (int *) malloc(size);
  for (int i = 0; i < N * N; i++) {
    model[i] = (rand() > RAND_MAX / 2) ? -1 : 1;
  }
  // printf("MODEL BEFORE FIRST ITERATION\n");
  // print_model(model, N);

  // The model before and after each iteration on the GPU.
  int *d_before, *d_after;
  cudaMalloc((void **)&d_before, size);
  cudaMalloc((void **)&d_after, size);
  cudaMemcpy(d_before, model, size, cudaMemcpyHostToDevice);

  struct timeval stop, start;
  gettimeofday(&start, NULL);

  for (int iter = 0; iter < K; iter++) {
    simulate_model<<<N * N, 1>>>(d_before, d_after, N);
    // Pass the `after` values to the `before` model for the next iteration.
    cudaMemcpy(d_before, d_after, size, cudaMemcpyDeviceToDevice);

    cudaMemcpy(after, d_after, size, cudaMemcpyDeviceToHost);
    // printf("\nMODEL AFTER ITERATION iter = %d\n", iter);
    // print_model(after, N);
  }

  gettimeofday(&stop, NULL);
  float timediff =
  (stop.tv_sec * 1000000.0 + (float)stop.tv_usec - start.tv_sec * 1000000.0 - (float)start.tv_usec) / 1000000;
  printf("\nV1: Size(%dx%d), iterations=%d, took %f seconds\n", N, N, K, timediff);
  
  FILE* results = fopen("results/v1_results.txt", "a");
  fprintf(results, "%d,%d,%f\n", N, K, timediff);
  fclose(results);

  return 0;
}

