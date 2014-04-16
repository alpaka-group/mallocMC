#pragma once

#include <cuda_runtime_api.h>
#include <string>

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

    static std::string classname(){
      return "CudaSetLimits";
    }

  };

} //namespace GetHeapPolicies
} //namespace PolicyMalloc
