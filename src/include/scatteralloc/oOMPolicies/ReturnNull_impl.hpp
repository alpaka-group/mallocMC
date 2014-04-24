#pragma once

#include <string>

#include "ReturnNull.hpp"

namespace PolicyMalloc{
namespace OOMPolicies{

  class ReturnNull
  {
    public:
      __device__ static void* handleOOM(void* mem){
        return NULL;
      }

      __host__ static std::string classname(){
        return "ReturnNull";
      }
  };

} //namespace OOMPolicies
} //namespace PolicyMalloc
