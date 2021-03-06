// -------------------------------------------------------------
// cuDPP -- CUDA Data Parallel Primitives library
// -------------------------------------------------------------
// $Revision: $
// $Date: $
// -------------------------------------------------------------
// This source code is distributed under the terms of license.txt in
// the root directory of this source distribution.
// -------------------------------------------------------------

/**
 * @file
 * test_spmvmult.cu
 *
 * @brief Host testrig routines to exercise cudpp's sparse matrix-vector functionality.
 */

#include <stdio.h>
#include <cutil.h>
#include <time.h>
#include <limits.h>

#include "cudpp.h"
#include "sparse.h"
#include "cudpp_testrig_options.h"

extern "C" void sparseMatrixVectorMultiplyGold(const MMMatrix * m, const float * x, float * y);
extern "C" void readMatrixMarket(MMMatrix * m, const char * filename);

/** just plain fopen, but if it doesn't work, lop off subdirectories 
 * until it does */
const char * findValidFile(const char * fileName)
{
    const char * fileNameTemp = fileName;
    FILE * inpFile;

    while (fileNameTemp[0])
    {
        inpFile = fopen(fileNameTemp, "r");
        if (inpFile)
        { 
            fclose(inpFile);
            return fileNameTemp;
        } 
        else
        {
            while (*fileNameTemp++ != '/')
                // effect: walks to the next path
                // "abc/def/efg" -> "def/efg" -> "efg" on successive iterations
                if (fileNameTemp[0] == '\0')
                {
                    // didn't find anything
                    return NULL;
                }
        }
    }
    return NULL;    
}


/**
 * testSparseMatrixVectorMultiply exercises cudpp's sparse matrix-vector functionality.
 * Possible command line arguments:
 * - --mat=filename: path to filename with matrix in MatrixMarket format
 * - Also "global" options (see setOptions)
 * @param argc Number of arguments on the command line, passed
 * directly from main
 * @param argv Array of arguments on the command line, passed directly
 * from main
 * @return Number of tests that failed regression (0 for all pass)
 * @see CUDPPConfiguration, setOptions, cudppScan
 */
int
testSparseMatrixVectorMultiply(int argc, const char** argv) 
{
    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));

    int retval = 0;

    testrigOptions testOptions;
    setOptions(argc, argv, testOptions);

    char * mfile;

    cutGetCmdLineArgumentstr(argc, (const char**) argv, "mat", &mfile);
    if (mfile == NULL)
    {
        fprintf(stderr, "Error: Must specify matrix with --mat=MATNAME\n");
        exit(1);
    }

    const char * foundMfile = findValidFile(mfile);
    if (!foundMfile)
    {
        fprintf(stderr, "Error: Unable to find file %s\n", mfile);
        exit(1);
    }

    MMMatrix m;
    readMatrixMarket(&m, foundMfile);

    const unsigned int cols = m.getCols();
    const unsigned int rows = m.getRows();
    const unsigned int entries = m.getNumEntries();

    printf("Rows = %d Cols = %d Non-zero entries = %d\n", rows, cols, entries);
    
    float * reference = (float *) malloc(sizeof(float) * rows);
    float * A = (float *) malloc(sizeof(float) * entries);
    unsigned int * indx = 
        (unsigned int *) malloc(sizeof(unsigned int) * entries);
    float * y = (float *) malloc(sizeof(float) * rows);
    float * x = (float *) malloc(sizeof(float) * cols);

    for (unsigned int i = 0; i < cols; i++)
    {
        x[i] = 1.0f;
    }

    for (unsigned int i = 0; i < rows; i++)
    {
        y[i] = 0.0f;
        reference[i] = 0.0f;
    }

    for (unsigned i = 0; i < entries; i++)
    {
        A[i] = m[i].getEntry();
        indx[i] = m[i].getCol();
    }

    // allocate device memory input, output, and temp arrays
    float * d_y;
    float * d_x;

    CUDA_SAFE_CALL(cudaMalloc((void**) &d_x, cols * sizeof(float))); 
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_y, rows * sizeof(float))); 

    CUDA_SAFE_CALL(cudaMemcpy(d_x, x, cols * sizeof(float),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_y, y, rows * sizeof(float),
                              cudaMemcpyHostToDevice));

    CUDPPConfiguration config;
    config.datatype = CUDPP_FLOAT;
    config.options = (CUDPPOption)0;
    config.algorithm = CUDPP_SPMVMULT;

    CUDPPHandle sparseMatrixHandle;
    CUDPPResult result = CUDPP_SUCCESS;
    result = cudppSparseMatrix(&sparseMatrixHandle, config, entries, 
                               rows, (void *)A, m.getRowPtrs(), indx);

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error creating Sparse matrix object\n");
        return 1;
    }

    // Run it once to avoid timing startup overhead
    cudppSparseMatrixVectorMultiply(sparseMatrixHandle, d_y, d_x);
    
    // Compute gold comparison
    sparseMatrixVectorMultiplyGold(&m, x, reference);
    
    // cutStartTimer(timer);
    for (int i = 0; i < testOptions.numIterations; i++)
    {
        CUDA_SAFE_CALL(cudaMemcpy(d_y, y, rows * sizeof(float),
                                  cudaMemcpyHostToDevice));
        cutStartTimer(timer);
        cudppSparseMatrixVectorMultiply(sparseMatrixHandle, d_y, d_x);
        cudaThreadSynchronize();
        cutStopTimer(timer);
    }

    CUDA_SAFE_CALL(cudaMemcpy(y, d_y, rows * sizeof(float),
                              cudaMemcpyDeviceToHost));

    // epsilon is 0.001f, answer must be within epsilon
    CUTBoolean spmv_result = cutComparefe(reference, y, rows, 0.001f);
    retval += (CUTTrue == spmv_result) ? 0 : 1;

    if (testOptions.debug)
    {
        for (unsigned int i = 0; i < rows; i++)
        {
            printf("i: %d\tref: %f\ty: %f\n", i, reference[i], y[i]);
        }
    }

    printf("sparsemv test %s\n", 
           (CUTTrue == spmv_result) ? "PASSED" : "FAILED");
    printf("Average execution time: %f ms\n", 
           cutGetTimerValue(timer) / testOptions.numIterations);

    // count FLOPS: y <- y + Mx
    // one flop for each entry in matrix for multiply
    // summing up all rows is (entry - rows)
    // adding y to resulting vector is another rows
    // total: 2 * entries
    printf("FLOPS: %f FLOPS\n", 
           float(2 * entries) * 1000.0f /
           (cutGetTimerValue(timer) / testOptions.numIterations));
    fflush(stdout);

    result = cudppDestroySparseMatrix(sparseMatrixHandle);

    if (result != CUDPP_SUCCESS)
    {
        printf("Error destroying Sparse Matrix\n");
    }

    cutDeleteTimer(timer);

    free(reference);
    free(y);
    free(A);
    free(x);
    free(indx);

    CUDA_SAFE_CALL(cudaFree(d_x));
    CUDA_SAFE_CALL(cudaFree(d_y));

    return retval;
}
