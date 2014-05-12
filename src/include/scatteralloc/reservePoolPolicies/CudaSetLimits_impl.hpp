#pragma once

#include <cuda_runtime_api.h>
#include <string>

#include "CudaSetLimits.hpp"

namespace PolicyMalloc{
namespace ReservePoolPolicies{

  /**
   * @brief set CUDA internal heap for device-side malloc calls
   *
   * This ReservePoolPolicy is intended for use with CUDA capable accelerators
   * that support at least compute capability 2.0. It should be used in
   * conjunction with a CreationPolicy that actually requires the CUDA-internal
   * heap to be sized by calls to cudaDeviceSetLimit()
   */
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

} //namespace ReservePoolPolicies
} //namespace PolicyMalloc
