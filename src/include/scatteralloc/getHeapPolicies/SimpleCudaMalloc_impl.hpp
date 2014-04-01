#pragma once

#include "../policy_malloc_utils.hpp"

#include "SimpleCudaMalloc.hpp"

namespace PolicyMalloc{
namespace GetHeapPolicies{

  struct SimpleCudaMalloc{
    static void* getMemPool(size_t memsize){
      void* pool;
      SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc(&pool, memsize));
      return pool;
    }

    static void resetMemPool(void* p){
      SCATTERALLOC_CUDA_CHECKED_CALL(cudaFree(p));
    }

  };

} //namespace GetHeapPolicies
} //namespace PolicyMalloc
