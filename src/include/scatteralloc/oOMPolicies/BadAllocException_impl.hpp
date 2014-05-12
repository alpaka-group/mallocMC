#pragma once

#include <assert.h>
#include <string>

#include "BadAllocException.hpp"

namespace PolicyMalloc{
namespace OOMPolicies{

  /**
   * @brief Throws a std::bad_alloc exception on OutOfMemory
   *
   * This OOMPolicy will throw a std::bad_alloc exception, if the accelerator
   * supports it. Currently, Nvidia CUDA does not support any form of exception
   * handling, therefore handleOOM() does not have any effect on these
   * accelerators. Using this policy on other types of accelerators that do not
   * support exceptions results in undefined behaviour.
   */
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

    static std::string classname(){
      return "BadAllocException";
    }
  };

} //namespace OOMPolicies
} //namespace PolicyMalloc
