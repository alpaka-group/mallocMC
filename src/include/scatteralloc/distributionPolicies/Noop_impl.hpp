#pragma once

#include <boost/cstdint.hpp>
#include <string>

#include "Noop.hpp"

namespace PolicyMalloc{
namespace DistributionPolicies{
    
  class Noop 
  {
    typedef boost::uint32_t uint32;

    public:

    __device__ uint32 collect(uint32 bytes){
      return bytes;
    }

    __device__ void* distribute(void* allocatedMem){
      return allocatedMem;
    }

    static std::string classname(){
      return "Noop";
    }

  };

} //namespace DistributionPolicies
} //namespace PolicyMalloc
