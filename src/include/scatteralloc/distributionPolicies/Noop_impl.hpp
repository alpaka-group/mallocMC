#pragma once

#include <boost/cstdint.hpp>

#include "Noop.hpp"

namespace PolicyMalloc{
namespace DistributionPolicies{
    
  class Noop 
  {
    typedef boost::uint32_t uint32;

    public:

    __device__ uint32 gather(uint32 bytes){
      return bytes;
    }

    __device__ void* distribute(void* allocatedMem){
      return allocatedMem;
    }

  };

} //namespace DistributionPolicies
} //namespace PolicyMalloc
