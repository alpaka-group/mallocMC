#pragma once

#include <boost/cstdint.hpp>
#include <string>

#include "Noop.hpp"

namespace PolicyMalloc{
namespace AlignmentPolicies{

  /**
   * @brief a policy that does nothing
   *
   * This AlignmentPolicy will not perform any distribution, but only return
   * its input (identity function)
   */
  class Noop{
    typedef boost::uint32_t uint32;

    public:

    static boost::tuple<void*,size_t> alignPool(void* memory, size_t memsize){
      return boost::make_tuple(memory,memsize);
    }

    __device__ static uint32 applyPadding(uint32 bytes){
      return bytes;
    }

    static std::string classname(){
      return "Noop";
    }

  };

} //namespace AlignmentPolicies
} //namespace PolicyMalloc
