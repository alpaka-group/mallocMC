#pragma once

#include "ReturnNull.hpp"

namespace PolicyMalloc{
namespace OOMPolicies{

  class ReturnNull
  {
    public:
      __device__ static void* handleOOM(void* mem){
        return NULL;
      }
  };

} //namespace OOMPolicies
} //namespace PolicyMalloc
