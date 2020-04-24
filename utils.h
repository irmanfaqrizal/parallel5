#ifndef __UTILS_H__
#define __UTILS_H__

#define NBEXPERIMENTS    5
double experiments [NBEXPERIMENTS] ;

double* allocMatrix(int dimension);

void initMatrix(int dimension, double *A);
void initMatrixZero(int dimension, double *A);

double* createMatrixCopy(int dimension, double *A);

void sequentialMatrixMultiplication_REF(int dimension, double *A, double *B, double *C);

int checkMatricesEquality(int dimension, double *V1, double *V2);

void printMatrix(int dimension, double *A);

/* return the average time in cycles over the values stored in
 * experiments vector */
double average_time();



#endif
