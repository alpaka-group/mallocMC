#pragma once

#include <assert.h>

#include "ReturnNull.hpp"

namespace PolicyMalloc{
namespace OOMPolicies{

  class BadAllocException
  {
    public:
      __device__ static void* handleOOM(void* mem){
        assert(false);
        // TODO exception handling does not work on device!
        return NULL;
      }
  };

} //namespace OOMPolicies
} //namespace PolicyMalloc
