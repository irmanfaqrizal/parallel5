#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include "utils.h"
#include <string.h>

/* a matrix multiplication without locality (column-first)*/
void sequentialMatrixMultiplication(int dimension, double *A, double *B, double *C)
{
    int i = 0;
    int j = 0;
    int k = 0;

    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            for (k = 0; k < dimension; k++) {
                C[i + j * dimension] += A[i + k * dimension] * B[k + j*dimension];
            }
        }
    }
}

int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

/* parallel matrix multiplication */
void parallelMatrixMultiplication(int dimension, double *locA, double *locB, double *locC, int rank, int size, double *global_C)
{   
    // Set from which process we will receive B
    int receiveFrom = 0;
    if (rank == 0) {receiveFrom = (size-1)%size; }
    else {receiveFrom = rank - 1;}

    // Request and status
    MPI_Request request1, request2;
    MPI_Status status;

    // Compute number of rows for each process, then compute how many data (cell) for each process
    int r = dimension / size;
    int dataEachP = r * dimension;

    // Temporary send and receive to another process
    double *tempS = (double *)malloc(sizeof(double) * dataEachP);
    memcpy(tempS, locB, sizeof(double) * dataEachP);
    double *tempR = (double *)malloc(sizeof(double) * dataEachP);

    // Loop P (size) times
    for (int step = 0; step < size; step++) {

        // Send and receive B matrix (we have to wait for each of them finished)
        MPI_Isend(tempS, dataEachP, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD, &request1);
        MPI_Wait(&request1, &status);
        MPI_Irecv(tempR, dataEachP, MPI_DOUBLE, receiveFrom, 0, MPI_COMM_WORLD, &request2);
        MPI_Wait(&request2, &status);

        // Compute matrix C for every process
        int block = mod((rank-step), size);
        for (int l = 0; l < size; l++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < r; j++) {
                    for (int k = 0; k < r; k++) {
                        locC[(i*dimension) + (l*r+j)] = locC[(i*dimension) + (l*r+j)] + locA[(i*dimension) + (block*r+k)] * tempS [(k*dimension) + (l*r+j)];
                    }
                }
            }
        }
        // // To print the values in each iteration :
        // printf("Step : %d, block:%d. Proc %d/%d sendTo:%d, recvFrom:%d, A : %.1lf %.1lf %.1lf %.1lf"
        // ", B : %.1lf %.1lf %.1lf %.1lf, C : %.1lf %.1lf %.1lf %.1lf\n",
        //     step, block, rank, size, (rank+1)%size, receiveFrom, locA[0], locA[1], locA[2], locA[3],
        //     tempS[0], tempS[1], tempS[2], tempS[3], locC[0], locC[1], locC[2], locC[3]
        //     );

        // Copy tempR to send a new B in the next step
        memcpy(tempS, tempR, sizeof(double) * dataEachP);
    }
    
    // free the temps and then gather every rows in C to process 0
    free(tempS);
    free(tempR);
    MPI_Gather(locC, dataEachP, MPI_DOUBLE, global_C, dataEachP, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    // Variables
    unsigned int exp ;
    double *A, *B ,*C;
    double *A_check, *B_check ,*C_check;
    unsigned int mat_size=0;
    int my_rank;
    int w_size;
    double start=0, av=0;

    // Init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);

    // Check Arguments
    if(argc != 2) {
        printf("usage: %s matrix_size\n",argv[0]);
        MPI_Finalize();
        return 0;
    } else {
        mat_size = atoi(argv[1]);
    }

    // Allocate the matrix (for rank 0)
    if(my_rank == 0) {
        printf("test with a matrix of size %u x %u\n",mat_size, mat_size);
        A = allocMatrix(mat_size);
        B = allocMatrix(mat_size);
        C = allocMatrix(mat_size);
    }

    // Allocate rows in each process
    int dataEachP = mat_size*mat_size/w_size;
    double *local_A = (double *)malloc(sizeof(double) * dataEachP);
    double *local_B = (double *)malloc(sizeof(double) * dataEachP);
    double *local_C = (double *)malloc(sizeof(double) * dataEachP);

// #ifdef CHECK_CORRECTNESS
    /* running my sequential implementation of the matrix
       multiplication */
    if(my_rank == 0) {
        initMatrix(mat_size, A);
        initMatrix(mat_size, B);
        initMatrixZero(mat_size, C);
        A_check = createMatrixCopy(mat_size, A);
        B_check = createMatrixCopy(mat_size, B);
        C_check = allocMatrix(mat_size);
        initMatrixZero(mat_size, C_check);
    }

    // Distribute the matrix to all processes (scattered because a process needs portion of the matrix)
    MPI_Scatter( A, dataEachP, MPI_DOUBLE, local_A, dataEachP, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter( B, dataEachP, MPI_DOUBLE, local_B, dataEachP, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter( C, dataEachP, MPI_DOUBLE, local_C, dataEachP, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* check for correctness */
    if(my_rank == 0) {

        //sequentialMatrixMultiplication(mat_size, A, B , C);
        parallelMatrixMultiplication(mat_size, local_A, local_B, local_C, my_rank, w_size, C);
        sequentialMatrixMultiplication_REF(mat_size, A_check, B_check , C_check);
        printf("Result ");
        printMatrix(mat_size, C);
        printf("Expected ");
        printMatrix(mat_size, C_check);

        if(checkMatricesEquality(mat_size, C, C_check)) {
            printf("\t CORRECT matrix multiplication result \n");
        } else {
            printf("\t FAILED matrix multiplication !!! \n");
        }
        
        free(A_check);
        free(B_check);
        free(C_check);
    } else {
        parallelMatrixMultiplication(mat_size, local_A, local_B, local_C, my_rank, w_size, C);
    }

// #endif /* CHECK_CORRECTNESS */

    if(my_rank == 0) {
        free(A);
        free(B);
        free(C);
    }
    
    free (local_A);
    free (local_B);
    free (local_C);

    MPI_Finalize();
    return 0;
}
