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

#define N 5


void print_model(int **model, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%d ", model[i][j]);
    }
    printf("\n");
  }
}


int main (int argc, char **argv) {

  int **model = (int **) malloc(N * sizeof(int *));
  for (int i = 0; i < N; i++){
    model[i] = (int *) calloc(N, sizeof(int));
  }

  // The spins are initializes as 0's. We set them to +1 or -1 using a uniform distribution.
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      model[i][j] = (rand() > (int)RAND_MAX / 2) ? -1 : 1;
    }
  }

  return 0;
}
