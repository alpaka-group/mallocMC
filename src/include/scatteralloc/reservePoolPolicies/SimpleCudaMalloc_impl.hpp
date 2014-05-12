#pragma once

#include <string>

#include "../policy_malloc_utils.hpp"

#include "SimpleCudaMalloc.hpp"

namespace PolicyMalloc{
namespace ReservePoolPolicies{

  /**
   * @brief creates/allocates a fixed memory pool on the accelerator
   *
   * This ReservePoolPolicy will create a memory pool of a fixed size on the
   * accelerator by using a host-side call to cudaMalloc(). The pool is later
   * freed through cudaFree(). This can only be used with accelerators that
   * support CUDA and compute capability 2.0 or higher.
   */
  struct SimpleCudaMalloc{
    static void* setMemPool(size_t memsize){
      void* pool;
      SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc(&pool, memsize));
      return pool;
    }

    static void resetMemPool(void* p){
      SCATTERALLOC_CUDA_CHECKED_CALL(cudaFree(p));
    }

    static std::string classname(){
      return "SimpleCudaMalloc";
    }

  };

} //namespace ReservePoolPolicies
} //namespace PolicyMalloc
