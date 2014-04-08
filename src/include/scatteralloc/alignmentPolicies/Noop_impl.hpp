#pragma once

#include <boost/cstdint.hpp>

#include "Noop.hpp"

namespace PolicyMalloc{
namespace AlignmentPolicies{

  class Noop{
    typedef boost::uint32_t uint32;

    public:

    static void* alignPool(void* memory){
      return memory;
    }

    __device__ static uint32 alignAccess(uint32 bytes){
      return bytes;
    }

  };

} //namespace AlignmentPolicies
} //namespace PolicyMalloc
