#pragma once

#include <cuda_runtime_api.h>

#include "CudaSetLimits.hpp"

namespace PolicyMalloc{
namespace GetHeapPolicies{

  struct CudaSetLimits{
    static void* setMemPool(size_t memsize){
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, memsize);
      return NULL;
    }

    static void resetMemPool(void *p=NULL){
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8192U);
    }

  };

} //namespace GetHeapPolicies
} //namespace PolicyMalloc
