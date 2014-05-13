#pragma once

#include <boost/mpl/int.hpp>

namespace PolicyMalloc{
namespace AlignmentPolicies{

namespace ShrinkConfig{
  struct DefaultShrinkConfig{
    typedef boost::mpl::int_<16> dataAlignment;
  };
}

  /**
   * @brief Provides proper alignment of pool and pads memory requests
   *
   * This AlignmentPolicy is based on ideas from ScatterAlloc
   * (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604). It
   * performs alignment operations on big memory pools and requests to allocate
   * memory. Memory pools are truncated at the beginning until the pointer to
   * the memory fits the alignment. Requests to allocate memory are padded
   * until their size is a multiple of the alignment.
   *
   * @tparam T_Config (optional) The alignment to use
   */
  template<typename T_Config = ShrinkConfig::DefaultShrinkConfig>
  class Shrink;

} //namespace AlignmentPolicies
} //namespace PolicyMalloc
