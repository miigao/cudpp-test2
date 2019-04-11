#include "cudpp.h"
#include "cudpp_plan_manager.h"
#include "cudpp_scan.h"
#include "cudpp_segscan.h"
#include "cudpp_compact.h"
#include "cudpp_spmvmult.h"
#include "cudpp_radixsort.h"
#include "cudpp_rand.h"

/**
 * @brief Sorts key-value pairs or keys only
 * 
 * Takes as input an array of keys in GPU memory
 * (d_keys) and an optional array of corresponding values,
 * and outputs sorted arrays of keys and (optionally) values in place. 
 * Key-value and key-only sort is selected through the configuration of 
 * the plan, using the options CUDPP_OPTION_KEYS_ONLY and 
 * CUDPP_OPTION_KEY_VALUE_PAIRS.
 *
 * Supported key types are CUDPP_FLOAT and CUDPP_UINT.  Values can be
 * any 32-bit type (internally, values are treated only as a payload
 * and cast to unsigned int).
 *
 * @todo Determine if we need to provide an "out of place" sort interface.
 * 
 * @param[in] planHandle handle to CUDPPSortPlan
 * @param[out] d_keys keys by which key-value pairs will be sorted
 * @param[in] d_values values to be sorted
 * @param[in] keyBits the number of least significant bits in each element 
 *            of d_keys to sort by
 * @param[in] numElements number of elements in d_keys and d_values
 *
 * @see cudppPlan, CUDPPConfiguration, CUDPPAlgorithm
 */
CUDPP_DLL
CUDPPResult cudppSort(CUDPPHandle planHandle,
                      void        *d_keys,
                      void        *d_values,                      
                      int         keyBits,
                      size_t      numElements)
{
    CUDPPRadixSortPlan *plan = (CUDPPRadixSortPlan*)CUDPPPlanManager::GetPlan(planHandle);
    if (plan != NULL)
    {
        cudppRadixSortDispatch(d_keys, d_values, numElements, keyBits, plan);
        return CUDPP_SUCCESS;
    }
    else
    {
        return CUDPP_ERROR_UNKNOWN; //! @todo Return more specific errors.
    }
}
