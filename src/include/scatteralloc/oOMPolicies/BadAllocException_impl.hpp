#pragma once

#include <assert.h>

#include "BadAllocException.hpp"

namespace PolicyMalloc{
namespace OOMPolicies{

  struct BadAllocException
  {
    __device__ static void* handleOOM(void* mem){
#ifdef __CUDACC__
//#if __CUDA_ARCH__ < 350
#define PM_EXCEPTIONS_NOT_SUPPORTED_HERE
//#endif
#endif

#ifdef PM_EXCEPTIONS_NOT_SUPPORTED_HERE
#undef PM_EXCEPTIONS_NOT_SUPPORTED_HERE
      assert(false);
#else
      std::bad_alloc exception;
      throw exception;
#endif
      return mem;
    }
  };

} //namespace OOMPolicies
} //namespace PolicyMalloc
