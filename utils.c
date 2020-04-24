#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "utils.h"

double average_time (void)
{
    unsigned int i ;
    double s = 0 ;

    for (i = 0; i < NBEXPERIMENTS; i++)
    {
        s = s + experiments [i] ;
    }

    return s / NBEXPERIMENTS ;
}


/* allocates a n x n matrix */
double* allocMatrix(int dimension)
{
    double* mat = (double*)malloc(sizeof(double) * dimension * dimension);

    return mat;
}

/* initializes the content of matrix A */
void initMatrix(int dimension, double *A)
{
    int i = 0;
    int j = 0;

#ifdef RINIT
    srand(time(NULL));
#endif

    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
#ifdef RINIT
            A[i*dimension + j] = rand() % dimension;
#else
            A[i*dimension + j] = 2;
#endif
        }
    }
    
}


/* fills matrix A with 0*/
void initMatrixZero(int dimension, double *A)
{
    memset(A, 0, sizeof(double) * dimension * dimension);
}


/* create a copy of matrix A */
double* createMatrixCopy(int dimension, double *A)
{
    double* mat = (double*)malloc(sizeof(double) * dimension * dimension);

    memcpy(mat, A, sizeof(double) * dimension * dimension);
    
    return mat;
}



/* multiplication of square matrices of size dimension: C = A x B */
void sequentialMatrixMultiplication_REF(int dimension, double *A, double *B, double *C)
{
    int i = 0;
    int j = 0;
    int k = 0;

    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            for (k = 0; k < dimension; k++) {
                C[i*dimension + j] += A[i*dimension + k] * B[k*dimension + j];
            }
        }
    }
}

/* return 1 if V1 and V2 are equal */
int checkMatricesEquality(int dimension, double *V1, double *V2)
{
    int i = 0;
    int j = 0;
    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            if (V1[i*dimension + j] != V2[i*dimension + j]) {
                return 0;
            }
        }
    }
    return 1;
}

/* display the content of a matrix */
void printMatrix(int dimension, double *A)
{
    int i = 0;
    int j = 0;
    printf("Matrix content:\n");
    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            printf("\t %.1lf",A[i * dimension +j]);
        }
        printf("\n");
    }
    
}
