#pragma once

#include <boost/mpl/bool.hpp>
#include <boost/mpl/int.hpp>

namespace PolicyMalloc{
namespace CreationPolicies{
namespace ScatterConf{
  struct DefaultScatterConfig{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
  };

  struct DefaultScatterHashingParams{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
  };  
}

  /**
   * @brief fast memory allocation based on ScatterAlloc
   *
   * This CreationPolicy implements a fast memory allocator that trades speed
   * for fragmentation of memory. This is based on the memory allocator
   * "ScatterAlloc"
   * (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604), and
   * is extended to report free memory slots of a given size (both on host and
   * accelerator).
   * To work properly, this policy class requires a pre-allocated heap on the
   * accelerator and works only with Nvidia CUDA capable accelerators that have
   * at least compute capability 2.0.
   *
   * @tparam T_Config (optional) configure the heap layout. The
   *        default can be obtained through Scatter<>::HeapProperties
   * @tparam T_Hashing (optional) configure the parameters for
   *        the hashing formula. The default can be obtained through
   *        Scatter<>::HashingProperties
   */
  template<
  class T_Config = ScatterConf::DefaultScatterConfig,
  class T_Hashing = ScatterConf::DefaultScatterHashingParams
  >
  class Scatter;

}// namespace CreationPolicies
}// namespace PolicyMalloc
