#include <stdio.h>
#include "cutil.h"
#include <math.h>

#include "cudpp.h"
#include "cudpp_plan_manager.h"
#include "cudpp_radixsort.h"
#include "kernel/radixsort_kernel.cu"
#include "radixsort_app.cu"


int radixSortTest()
{
    int keybits = 32;
    unsigned *h_keys, *h_keysSorted, *d_keys;
    
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SORT_RADIX;
    config.datatype = CUDPP_UINT;
    config.options = CUDPP_OPTION_KEYS_ONLY;

    CUDPPHandle plan;
    cudppPlan(&plan, config, numElements, 1, 0);	

    h_keys       = (unsigned*)malloc(numElements*sizeof(unsigned));
    h_keysSorted = (unsigned*)malloc(numElements*sizeof(unsigned));

    // Fill up with some random data   
    //	mig

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_keys, numElements*sizeof(unsigned)));

    CUDA_SAFE_CALL(cudaMemcpy(d_keys, h_keys, numElements * sizeof(unsigned), cudaMemcpyHostToDevice));

    CUDPPRadixSortPlan *my_plan = (CUDPPRadixSortPlan*)CUDPPPlanManager::GetPlan(plan);

    radixSortKeysOnly((uint*)d_keys, my_plan, false, numElements, keybits);
    
    CUT_CHECK_ERROR("testradixSort - cudppRadixSort");

    // copy results
    CUDA_SAFE_CALL(cudaMemcpy(h_keysSorted, d_keys, numElements * sizeof(unsigned), cudaMemcpyDeviceToHost));

    //check results
    //  mig
    
    cudaFree(d_keys);
    free(h_keys);
    cudppDestroyPlan(plan);

    return 0;
}

int main()
{
    radixSortTest();
    return 0;
}
