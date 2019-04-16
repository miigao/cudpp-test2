#include "cudpp.h"
#include "cudpp_util.h"
#include "cudpp_radixsort.h"
#include "cudpp_scan.h"
#include "kernel/radixsort_kernel.cu"

#include "cutil.h"
#include <cstdlib>
#include <cstdio>
#include <assert.h>

typedef unsigned int uint;

/** @brief Perform one step of the radix sort.  Sorts by nbits key bits per step, 
 * starting at startbit.
 * 
 * @param[in,out] keys  Keys to be sorted.
 * @param[in] plan Configuration information for RadixSort.
 * @param[in] numElements Number of elements in the sort. 
**/
template<uint nbits, uint startbit, bool flip, bool unflip>
void radixSortStepKeysOnly(uint *keys, 
                           const CUDPPRadixSortPlan *plan,
                           uint numElements)
{
    const uint eltsPerBlock = SORT_CTA_SIZE * 4;
    const uint eltsPerBlock2 = SORT_CTA_SIZE * 2;

    bool fullBlocks = ((numElements % eltsPerBlock) == 0);
    uint numBlocks = (fullBlocks) ? 
        (numElements / eltsPerBlock) : 
    (numElements / eltsPerBlock + 1);
    uint numBlocks2 = ((numElements % eltsPerBlock2) == 0) ?
        (numElements / eltsPerBlock2) : 
    (numElements / eltsPerBlock2 + 1);

    bool loop = numBlocks > 65535;
    
    uint blocks = loop ? 65535 : numBlocks;
    uint blocksFind = loop ? 65535 : numBlocks2;
    uint blocksReorder = loop ? 65535 : numBlocks2;

    uint threshold = fullBlocks ? plan->m_persistentCTAThresholdFullBlocks[1] : plan->m_persistentCTAThreshold[1];

    bool persist = plan->m_bUsePersistentCTAs && (numElements >= threshold);

    if (persist)
    {
        loop = (numElements > 262144) || (numElements >= 32768 && numElements < 65536);
        
        blocks = numBlocks;
        blocksFind = numBlocks2;
        blocksReorder = numBlocks2;
    }

    if (fullBlocks)
    {
        if (loop)
        {
            if (persist) 
            {
                blocks = flip ? numCTAs(radixSortBlocksKeysOnly<4, 0, true, true, true>) : 
                                numCTAs(radixSortBlocksKeysOnly<4, 0, true, false, true>);
            }

            radixSortBlocksKeysOnly<nbits, startbit, true, flip, true>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)keys, numElements, numBlocks);
        }
        else
            radixSortBlocksKeysOnly<nbits, startbit, true, flip, false>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)keys, numElements, numBlocks);
    }
    else
    {
        if (loop)
        {
            if (persist) 
            {
                blocks = flip ? numCTAs(radixSortBlocksKeysOnly<4, 0, false, true, true>) : 
                                numCTAs(radixSortBlocksKeysOnly<4, 0, false, false, true>);
            }

            radixSortBlocksKeysOnly<nbits, startbit, false, flip, true>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)keys, numElements, numBlocks);
        }
        else
            radixSortBlocksKeysOnly<nbits, startbit, false, flip, false>
                <<<blocks, SORT_CTA_SIZE, 4 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint4*)plan->m_tempKeys, (uint4*)keys, numElements, numBlocks);

    }

    if (fullBlocks)
    {
        if (loop)
        {
            if (persist) 
            {
                blocksFind = numCTAs(findRadixOffsets<0, true, true>);
            }
            findRadixOffsets<startbit, true, true>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
        }
        else
            findRadixOffsets<startbit, true, false>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
    }
    else
    {
        if (loop)
        {
            if (persist) 
            {
                blocksFind = numCTAs(findRadixOffsets<0, false, true>);
            }
            findRadixOffsets<startbit, false, true>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);
        }
        else
            findRadixOffsets<startbit, false, false>
                <<<blocksFind, SORT_CTA_SIZE, 3 * SORT_CTA_SIZE * sizeof(uint)>>>
                ((uint2*)plan->m_tempKeys, plan->m_counters, plan->m_blockOffsets, numElements, numBlocks2);

    }

    cudppScanDispatch(plan->m_countersSum, plan->m_counters, 16*numBlocks2, 1, plan->m_scanPlan);

    if (fullBlocks)
    {
        if (plan->m_bManualCoalesce)
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ? 
                        numCTAs(reorderDataKeysOnly<0, true, true, true, true>) : 
                        numCTAs(reorderDataKeysOnly<0, true, true, false, true>);
                }
                reorderDataKeysOnly<startbit, true, true, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                    numElements, numBlocks2);
            }
            else
                reorderDataKeysOnly<startbit, true, true, unflip, false>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                     numElements, numBlocks2);
        }
        else
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ?
                        numCTAs(reorderDataKeysOnly<0, true, false, true, true>) :
                        numCTAs(reorderDataKeysOnly<0, true, false, false, true>);
                }
                reorderDataKeysOnly<startbit, true, false, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                    numElements, numBlocks2);
            }
            else
                reorderDataKeysOnly<startbit, true, false, unflip, false>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                     numElements, numBlocks2);
        }
    }
    else
    {
        if (plan->m_bManualCoalesce)
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ? 
                        numCTAs(reorderDataKeysOnly<0, false, true, true, true>) :
                        numCTAs(reorderDataKeysOnly<0, false, true, false, true>);
                }
                reorderDataKeysOnly<startbit, false, true, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                    numElements, numBlocks2);
            }
            else
                reorderDataKeysOnly<startbit, false, true, unflip, false>
                <<<blocksReorder, SORT_CTA_SIZE>>>
                (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                numElements, numBlocks2);
        }
        else
        {
            if (loop)
            {
                if (persist) 
                {
                    blocksReorder = unflip ?
                        numCTAs(reorderDataKeysOnly<0, false, false, true, true>) :
                        numCTAs(reorderDataKeysOnly<0, false, false, false, true>);
                }
                reorderDataKeysOnly<startbit, false, false, unflip, true>
                    <<<blocksReorder, SORT_CTA_SIZE>>>
                    (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                    numElements, numBlocks2);
            }
            else
                reorderDataKeysOnly<startbit, false, false, unflip, false>
                <<<blocksReorder, SORT_CTA_SIZE>>>
                (keys, (uint2*)plan->m_tempKeys, plan->m_blockOffsets, plan->m_countersSum, plan->m_counters, 
                numElements, numBlocks2);
        }
    }

    CUT_CHECK_ERROR("radixSortStepKeysOnly");
}

/** 
 * @brief Main radix sort function. For keys only configuration.
 *
 * Main radix sort function.  Sorts in place in the keys array,
 * but uses the other device arrays as temporary storage.  All pointer 
 * parameters are device pointers.  Uses scan for the prefix sum of
 * radix counters.
 * 
 * @param[in,out] keys Keys to be sorted.
 * @param[in] plan Configuration information for RadixSort.
 * @param[in] flipBits Is set true if key datatype is a float (neg. numbers) 
 *        for special float sorting operations.
 * @param[in] numElements Number of elements in the sort.
 * @param[in] keyBits Number of interesting bits in the key
**/
extern "C"
void radixSortKeysOnly(uint *keys,
                       const CUDPPRadixSortPlan *plan, 
                       bool flipBits, 
                       size_t numElements,
                       int keyBits)
{

    if(numElements <= WARP_SIZE)
    {
        if (flipBits)
            radixSortSingleWarpKeysOnly<true><<<1, numElements>>>(keys, numElements);
        else
            radixSortSingleWarpKeysOnly<false><<<1, numElements>>>(keys, numElements);
        return;
    }
    if(numElements <= SORT_CTA_SIZE * 4)
    {
        if (flipBits)
            radixSortSingleBlockKeysOnly<true>(keys, numElements);
        else
            radixSortSingleBlockKeysOnly<false>(keys, numElements);
        return;
    }

    // flip float bits on the first pass, unflip on the last pass
    if (flipBits) 
    {
        radixSortStepKeysOnly<4,  0, true, false>(keys, plan, numElements);
    }
    else
    {
        radixSortStepKeysOnly<4,  0, false, false>(keys, plan, numElements);
    }

    if (keyBits > 4)
    {
        radixSortStepKeysOnly<4,  4, false, false>(keys, plan, numElements);
    }
    if (keyBits > 8)
    {
        radixSortStepKeysOnly<4,  8, false, false>(keys, plan, numElements);
    }
    if (keyBits > 12)
    {
        radixSortStepKeysOnly<4, 12, false, false>(keys, plan, numElements);
    }
    if (keyBits > 16)
    {
        radixSortStepKeysOnly<4, 16, false, false>(keys, plan, numElements);
    }
    if (keyBits > 20)
    {
        radixSortStepKeysOnly<4, 20, false, false>(keys, plan, numElements);
    }
    if (keyBits > 24)
    {
       radixSortStepKeysOnly<4, 24, false, false>(keys, plan, numElements);
    }
    if (keyBits > 28)
    {
        if (flipBits) // last pass
        {
            radixSortStepKeysOnly<4, 28, false, true>(keys, plan, numElements);
        }
        else
        {
            radixSortStepKeysOnly<4, 28, false, false>(keys, plan, numElements);
        }
    }
}

extern "C"
void initDeviceParameters(CUDPPRadixSortPlan *plan)
{
    int deviceID = -1;
    if (cudaSuccess == cudaGetDevice(&deviceID))
    {
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, deviceID);

        int smVersion = devprop.major * 10 + devprop.minor;

        // sm_12 and later devices don't need help with coalesce in reorderData kernel
        plan->m_bManualCoalesce = (smVersion < 12);

        // sm_20 and later devices are better off not using persistent CTAs
        plan->m_bUsePersistentCTAs = (smVersion < 20);

        if (plan->m_bUsePersistentCTAs)
        {
            // The following is only true on pre-sm_20 devices (pre-Fermi):
            // Empirically we have found that for some (usually larger) sort
            // sizes it is better to use exactly as many "persistent" CTAs 
            // as can fill the GPU, which loop over the "blocks" of work. For smaller 
            // arrays it is better to use the typical CUDA approach of launching one CTA
            // per block of work.
            // 0-element of these two-element arrays is for key-value sorts
            // 1-element is for key-only sorts
            plan->m_persistentCTAThreshold[0] = plan->m_bManualCoalesce ? 16777216 : 524288;
            plan->m_persistentCTAThresholdFullBlocks[0] = plan->m_bManualCoalesce ? 2097152: 524288;
            plan->m_persistentCTAThreshold[1] = plan->m_bManualCoalesce ? 16777216 : 8388608;
            plan->m_persistentCTAThresholdFullBlocks[1] = plan->m_bManualCoalesce ? 2097152: 0;

            // create a map of function pointers to register counts for more accurate occupancy calculation
            // Must pass in the dynamic shared memory used by each kernel, since the runtime doesn't know it
            // Note we only insert the "loop" version of the kernels (the one with the last template param = true)
            // Because those are the only ones that require persistent CTAs that maximally fill the device.
            computeNumCTAs(radixSortBlocks<4, 0, false, false, true>,         4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            computeNumCTAs(radixSortBlocks<4, 0, false, true,  true>,         4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            computeNumCTAs(radixSortBlocks<4, 0, true, false,  true>,         4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            computeNumCTAs(radixSortBlocks<4, 0, true, true,  true>,          4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            
            computeNumCTAs(radixSortBlocksKeysOnly<4, 0, false, false, true>, 4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            computeNumCTAs(radixSortBlocksKeysOnly<4, 0, false, true, true>,  4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            computeNumCTAs(radixSortBlocksKeysOnly<4, 0, true, false, true>,  4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            computeNumCTAs(radixSortBlocksKeysOnly<4, 0, true, true, true>,   4 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);

            computeNumCTAs(findRadixOffsets<0, false, true>,                  3 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);
            computeNumCTAs(findRadixOffsets<0, true, true>,                   3 * SORT_CTA_SIZE * sizeof(uint), SORT_CTA_SIZE);

            computeNumCTAs(reorderData<0, false, false, false, true>,         0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderData<0, false, false, true, true>,          0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderData<0, false, true, false, true>,          0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderData<0, false, true, true, true>,           0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderData<0, true, false, false, true>,          0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderData<0, true, false, true, true>,           0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderData<0, true, true, false, true>,           0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderData<0, true, true, true, true>,            0,                                SORT_CTA_SIZE);

            computeNumCTAs(reorderDataKeysOnly<0, false, false, false, true>, 0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderDataKeysOnly<0, false, false, true, true>,  0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderDataKeysOnly<0, false, true, false, true>,  0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderDataKeysOnly<0, false, true, true, true>,   0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderDataKeysOnly<0, true, false, false, true>,  0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderDataKeysOnly<0, true, false, true, true>,   0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderDataKeysOnly<0, true, true, false, true>,   0,                                SORT_CTA_SIZE);
            computeNumCTAs(reorderDataKeysOnly<0, true, true, true, true>,    0,                                SORT_CTA_SIZE);
                   
            computeNumCTAs(emptyKernel,                                       0,                                SORT_CTA_SIZE);
        }
    }
}

/**
 * @brief From the programmer-specified sort configuration, 
 *        creates internal memory for performing the sort.
 * 
 * @param[in] plan Pointer to CUDPPRadixSortPlan object
**/
extern "C"
void allocRadixSortStorage(CUDPPRadixSortPlan *plan)
{               
        
    unsigned int numElements = plan->m_numElements;

    unsigned int numBlocks = 
        ((numElements % (SORT_CTA_SIZE * 4)) == 0) ? 
            (numElements / (SORT_CTA_SIZE * 4)) : 
            (numElements / (SORT_CTA_SIZE * 4) + 1);
                        
    switch(plan->m_config.datatype)
    {
    case CUDPP_UINT:
        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_tempKeys, 
                                  numElements * sizeof(unsigned int)));

        if (!plan->m_bKeysOnly)
            CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_tempValues, 
                           numElements * sizeof(unsigned int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_counters, 
                       WARP_SIZE * numBlocks * sizeof(unsigned int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_countersSum,
                       WARP_SIZE * numBlocks * sizeof(unsigned int)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_blockOffsets, 
                       WARP_SIZE * numBlocks * sizeof(unsigned int)));
    break;

    case CUDPP_FLOAT:
        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_tempKeys,
                                   numElements * sizeof(float)));

        if (!plan->m_bKeysOnly)
            CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_tempValues,
                           numElements * sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_counters,
                       WARP_SIZE * numBlocks * sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_countersSum,
                       WARP_SIZE * numBlocks * sizeof(float)));

        CUDA_SAFE_CALL(cudaMalloc((void **)&plan->m_blockOffsets,
                       WARP_SIZE * numBlocks * sizeof(float)));     
    break;
    }
        
    initDeviceParameters(plan);
}

/** @brief Deallocates intermediate memory from allocRadixSortStorage.
 *
 *
 * @param[in] plan Pointer to CUDPPRadixSortPlan object
**/
extern "C"
void freeRadixSortStorage(CUDPPRadixSortPlan* plan)
{
    CUDA_SAFE_CALL( cudaFree(plan->m_tempKeys));
    CUDA_SAFE_CALL( cudaFree(plan->m_tempValues));
    CUDA_SAFE_CALL( cudaFree(plan->m_counters));
    CUDA_SAFE_CALL( cudaFree(plan->m_countersSum));
    CUDA_SAFE_CALL( cudaFree(plan->m_blockOffsets));
}
