/**
 * @file: v0_sequential.c
 * ****************************************
 * @author: Antonios Antoniou
 * @email: aantonii@ece.auth.gr
 * ****************************************
 * @description: Simulate the Ising model for a system of size `n x n` and `k` iterations.
 * Uses the CPU and implements it sequentially.
 * ****************************************
 * Parallel and Distributed Systems - Electrical and Computer Engineering
 * 2022 Aristotle University Thessaloniki
 */ 
#include <stdio.h>
#include <stdlib.h>


void print_model(int **model, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%d  ", model[i][j]);
    }
    printf("\n");
  }
}


// Implement the sign function with a node's neighbours.
int sign(int self, int *neighbours, int neighbours_n) {
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


/**
 * Instead of using if's or implementing a struct of any sort, we will be using this.
 * Rolls the array into itself. Index `N` points to `0` and index `-1` points to `N-1`
 * Ask for the indices and the size of the model. 
 * `+N` takes care of negative indices,
 * `%N` takes care of indices greater than the size of the array.
 */
int get_model(int **model, int i, int j, int n) {
  return model[(i + n) % n][(j + n) % n];
}


int main (int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: v0.out N K, where N is size and K is iterations\n");
    return -1;
  }
  int N = atoi(argv[1]);
  int K = atoi(argv[2]);

  int **model = (int **) malloc(N * sizeof(int *));
  for (int i = 0; i < N; i++){
    model[i] = (int *) calloc(N, sizeof(int));
  }

  // The spins are initialized as 0's. We set them to +1 or -1 using a uniform distribution.
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      model[i][j] = (rand() > RAND_MAX / 2) ? -1 : 1;
    }
  }
  printf("MODEL BEFORE FIRST ITERATION\n");
  print_model(model, N);

  /* ---------- START SIMULATION ---------- */
  int neighbours[4];
  // The model before and after an iteration.
  int **before = (int **) malloc(N * sizeof(int *));
  int **after = (int **) malloc(N * sizeof(int *));

  for (int i = 0; i < N; i++) {
    before[i] = (int *) malloc(N * sizeof(int));
    after[i] = (int *) calloc(N, sizeof(int));
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      before[i][j] = model[i][j];
    }
  }

  for (int iter = 0; iter < K; iter++) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        neighbours[0] = get_model(before, i, j + 1, N);
        neighbours[1] = get_model(before, i, j - 1, N);
        neighbours[2] = get_model(before, i + 1, j, N);
        neighbours[3] = get_model(before, i - 1, j, N);

        after[i][j] = sign(before[i][j], neighbours, 4);
      }
    }

    // After an iteration is done, pass the `after` values to the `before` model.
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        before[i][j] = after[i][j];
      }
    }

    printf("\nMODEL ON ITERATION iter = %d\n", iter);
    print_model(after, N);
  }

  return 0;
}
