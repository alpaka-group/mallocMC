#pragma once

#include <boost/cstdint.hpp>

#include "OldMalloc.hpp"

namespace PolicyMalloc{
namespace CreationPolicies{
    
  class OldMalloc
  {
    typedef boost::uint32_t uint32;

    public:
    __device__ void* create(uint32 bytes)
    {
      return ::malloc(static_cast<size_t>(bytes));
    }

    __device__ void destroy(void* mem)
    {
      free(mem);
    }

    __device__ bool isOOM(void* p){
      return  32 == __popc(__ballot(p == NULL));
    }

    template < typename T>
    static void* initHeap(const T& obj, void* pool, size_t memsize){
      T* dAlloc;
      SCATTERALLOC_CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&dAlloc,obj));
      return dAlloc;
    }   

    template < typename T>
    static void finalizeHeap(const T& obj){
    }

  };

} //namespace CreationPolicies
} //namespace PolicyMalloc
