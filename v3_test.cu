#include <stdio.h>
#include <stdlib.h>


void print_model(int *model, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      printf("%d ", model[i * size + j]);
    }
    printf("\n");
  }
}


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
__global__ void simulate_model(int *before, int *after, int size) {
  b = blockDim.x /* x and y are identical */
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int neighbours[4];

  // The northwest position of each block. Iterate through the rest of the moments.
  int index = (i * size + j) * b;

  // The part of the model represented by threads in the block.
  __shared__ int* block_model; 
  // Increase size to include the outside nodes' neighbours.
  int model_size = (blockDim.x + 2) * (blockDim.y + 2);
  block_model = (int*) malloc(model_size * sizeof(int));
  block_model[(threadIdx.x + 1) * blockDim.y + threadIdx.y + 1] = before[index];

  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    for (int i = 0; i < blockDim.x + 2; i++) {
      for (int j = 0; j < blockDim.y + 2; j++) {
        printf("%d ", block_model[i * blockDim.y + j]);
      }
      printf("\n");
    }
  }

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


int main(int argc, char **argv) {
  const int N = 16;
  const int BLOCKSIZE = 4;
  const int K = 1;
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
  // Divide the block twice. Raise fewer threads that produce more workload.
  dim3 dim_grid(N / dim_block.x, N / dim_block.y);

  for (int iter = 0; iter < K; iter++) {
    simulate_model<<<dim_grid, dim_block>>>(d_before, d_after, N);
    // Pass the `after` values to the `before` model for the next iteration.
    cudaMemcpy(d_before, d_after, size, cudaMemcpyDeviceToDevice);

    cudaMemcpy(after, d_after, size, cudaMemcpyDeviceToHost);
    printf("\nMODEL AFTER ITERATION iter = %d\n", iter);
    //print_model(after, N);
  }

  return 0;
}