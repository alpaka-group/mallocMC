#pragma once

#include <string>

#include "../policy_malloc_utils.hpp"

#include "SimpleCudaMalloc.hpp"

namespace PolicyMalloc{
namespace ReservePoolPolicies{

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
