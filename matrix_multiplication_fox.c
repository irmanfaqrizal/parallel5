#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include "utils.h"
#include <string.h>
#include <stdbool.h> 

typedef struct {
    MPI_Comm  grid;                  // Communicator for entire grid
    MPI_Comm  row;                   // Communicator for specific row in the grid 
    MPI_Comm  col;                   // Communicator for specific column in the grid
    int       rowPosition;           // local row coordinate
    int       columnPosition;        // local column coordinate
    int       gridRank;              // rank for each processor
} COMMS;

void initCommunicators(COMMS* comms, int w_size) {
    int dimensions[2];
    int period[2];
    int coordinates[2];
    int freeCoordinates[2];
    dimensions[0] = dimensions[1] = (int) sqrt(w_size);
    period[0] = period[1] = 1;

    // We first create the global communicator, then we assign a rank to each processor
    // Use this two to get the coordinates for each processor (rows and column)
    MPI_Cart_create(MPI_COMM_WORLD, 2, dimensions, period, 1, &(comms->grid));
    MPI_Comm_rank(comms->grid, &(comms->gridRank));
    MPI_Cart_coords(comms->grid, comms->gridRank, 2, coordinates);
    comms->rowPosition = coordinates[0];
    comms->columnPosition = coordinates[1];

    // Assign communicator for rows and cols
    freeCoordinates[0] = 0; 
    freeCoordinates[1] = 1;
    MPI_Cart_sub(comms->grid, freeCoordinates, &(comms->row));
    freeCoordinates[0] = 1; 
    freeCoordinates[1] = 0;
    MPI_Cart_sub(comms->grid, freeCoordinates, &(comms->col));
}

/* fox matrix multiplication */
void foxMatrixMultiplication(COMMS *comms, int dimension,
    double *locA, double *locB, double *locC, int rank, int size, double *global_C)
{   
    int recvSource;		
	int sendDestination;		
	int root = 0;
	bool isOnDiagonal = false;
	MPI_Status status;

	// size of a block is a dimension of the number of process
	int sqroot = (int) sqrt(size);

	// Compute the source and destination to shift the blocks in earch iteration
	recvSource = (comms->rowPosition + 1) % sqroot;
	sendDestination = (comms->rowPosition - 1 + sqroot ) % sqroot;
	
	// Temporary matrices : init with zeros
	double* temp_A = malloc(size * sizeof(double));
	double* temp_B = malloc(size * sizeof(double));
	int i;
	for(i=0; i < size; ++i) {
		temp_A[i] = 0.;
		temp_B[i] = 0.;
	}

	// Make sure every process initialized before start the algorithm
	MPI_Barrier(comms->row);		
	
    // start iterations
	for(i = 0; i < sqroot; i++ ) {

        // Broadcast block A
		root = (comms->rowPosition + i)%sqroot;
		if( root == comms->columnPosition ){
			MPI_Bcast( locA, size, MPI_DOUBLE, root, comms->row);
			isOnDiagonal = true;	
		} else{
			MPI_Bcast(temp_A, size, MPI_DOUBLE, root, comms->row );
			isOnDiagonal = false;
		}
		
        // Multply block A with B if is on diagonal, and temp A with B if its not
        int ind, j, k;
        if (isOnDiagonal) {
            for (ind = 0; ind < sqroot; ind++)
                for (j = 0; j < sqroot; j++)
                        for (k = 0; k < sqroot; k++)
                            locC[ind*sqroot+j] += locA[ind*sqroot+k]*locB[k*sqroot+j];
        } else {
            for (ind = 0; ind < sqroot; ind++)
                for (j = 0; j < sqroot; j++)
                        for (k = 0; k < sqroot; k++)
                            locC[ind*sqroot+j] += temp_A[ind*sqroot+k]*locB[k*sqroot+j];
        }
	
        // shift the blocks vertically for next iteration
		MPI_Send( locB, size, MPI_DOUBLE, sendDestination, 0, comms->col);
		MPI_Recv( temp_B, size, MPI_DOUBLE, recvSource, 0,comms->col, &status);
		int z;
		for(z = 0; z < size; ++z)
			locB[z] = temp_B[z]; 
	}
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

    // Check Arguments
    if(argc != 2) {
        printf("usage: %s matrix_size\n",argv[0]);
        return 0;
    } else {
        mat_size = atoi(argv[1]);
    }

    // Init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);

    // Init grid informations and communicators
    COMMS comms;
    initCommunicators(&comms, w_size);

    // Allocate the matrix (for rank 0)
    if(my_rank == 0) {
        printf("test with a matrix of size %u x %u\n",mat_size, mat_size);
        A = allocMatrix(mat_size);
        B = allocMatrix(mat_size);
        C = allocMatrix(mat_size);
    }

    // Setting parameters for creating a new datatype
    MPI_Datatype blocktype, type;	
	int arraySize[2] = {w_size, w_size};
	int subarraySizes[2] = {(int)sqrt(w_size), (int) sqrt(w_size)};
	int arrayStart[2] = {0,0};

    // the array size will be number of process x number of process, and sub arrays are the square roots of them
    // 'blocktype' will return a new datatype, but then we resized this into 'type'
    // then we have to commit using MPI_Type_commit
	MPI_Type_create_subarray(2, arraySize, subarraySizes, arrayStart, MPI_ORDER_C, MPI_DOUBLE, &blocktype); 
	MPI_Type_create_resized(blocktype, 0, (int)sqrt(w_size)*sizeof(double), &type);
	MPI_Type_commit(&type);

    // Allocate blocks in each process
    int dataEachP = mat_size*mat_size/w_size;
    double *local_A = (double *)malloc(sizeof(double) * dataEachP);
    double *local_B = (double *)malloc(sizeof(double) * dataEachP);
    double *local_C = (double *)malloc(sizeof(double) * dataEachP);

    // Set up the parameters for scatterv
    // sendcounts contains an array of how much data will be sent for each process
    // displs will specify from which part of the global data (process 0) another process will take
	int sendcounts[w_size];
    int displs[w_size];
    int i, j;
    if (my_rank == 0) {
		for(i=0; i<w_size; i++)  {
			sendcounts[i] = 1;
		}

		int disp = 0;
		for (i=0; i<(int)sqrt(w_size); i++) {
			for (j=0; j<(int)sqrt(w_size); j++) {
				displs[i*(int)sqrt(w_size)+j] = disp;
				disp += 1;
			}
			disp += ((w_size/(int)sqrt(w_size)-1))*(int)sqrt(w_size);
		}
	}

    // init matrices
    if(my_rank == 0) {
        initMatrix(mat_size, A);
        initMatrix(mat_size, B);
        initMatrixZero(mat_size, C);
        A_check = createMatrixCopy(mat_size, A);
        B_check = createMatrixCopy(mat_size, B);
        C_check = allocMatrix(mat_size);
        initMatrixZero(mat_size, C_check);
    }

    // Scatter to all processes
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, local_A, w_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	MPI_Scatterv(B, sendcounts, displs, MPI_DOUBLE, local_B, w_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Run the function and gather the result
    foxMatrixMultiplication(&comms, mat_size, local_A, local_B, local_C, my_rank, w_size, C);
    MPI_Gatherv(local_C, w_size,  MPI_DOUBLE, C, sendcounts, displs, type, 0, MPI_COMM_WORLD);

    /* check for correctness */
    if(my_rank == 0) {   
        sequentialMatrixMultiplication_REF(mat_size, A_check, B_check , C_check);
        printMatrix(mat_size, C);
        printMatrix(mat_size, C_check);
        if(checkMatricesEquality(mat_size, C, C_check)) {
            printf("\t CORRECT matrix multiplication result \n");
        } else {
            printf("\t FAILED matrix multiplication !!! \n");
        }
        
        free(A_check);
        free(B_check);
        free(C_check);
    }

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
