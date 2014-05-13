#pragma once

#include <boost/mpl/int.hpp>

namespace PolicyMalloc{
namespace DistributionPolicies{
    
  namespace XMallocSIMDConf{
    struct DefaultXMallocConfig{
      typedef boost::mpl::int_<4096>     pagesize;
    };  
  }

  /**
   * @brief SIMD optimized chunk resizing in the style of XMalloc
   *
   * This DistributionPolicy can take the memory requests from a group of
   * worker threads and combine them, so that only one of the workers will
   * allocate the whole request. Later, each worker gets an appropriate offset
   * into the allocated chunk. This is beneficial for SIMD architectures since
   * only one of the workers has to compete for the resource.  This algorithm
   * is inspired by the XMalloc memory allocator
   * (http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5577907&tag=1) and
   * its implementation in ScatterAlloc
   * (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604)
   * XMallocSIMD is inteded to be used with Nvidia CUDA capable accelerators
   * that support at least compute capability 2.0
   *
   * @tparam T_Config (optional) The configuration struct to overwrite
   *        default configuration. The default can be obtained through
   *        XMallocSIMD<>::Properties
   */
  template<class T_Config=XMallocSIMDConf::DefaultXMallocConfig>
  class XMallocSIMD;


} //namespace DistributionPolicies
} //namespace PolicyMalloc
