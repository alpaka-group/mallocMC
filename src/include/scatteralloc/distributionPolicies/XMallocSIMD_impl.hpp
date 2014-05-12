#pragma once

#include <boost/cstdint.hpp>
#include <boost/static_assert.hpp>
#include <limits>
#include <string>
#include <sstream>

#include "../policy_malloc_utils.hpp"
#include "XMallocSIMD.hpp"

namespace PolicyMalloc{
namespace DistributionPolicies{

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
  template<class T_Config>
  class XMallocSIMD
  {
    private:

      typedef boost::uint32_t uint32;
      bool can_use_coalescing;
      uint32 warpid;
      uint32 myoffset;
      uint32 threadcount;
      uint32 req_size;
    public:
      typedef T_Config Properties;

    private:
/** Allow for a hierarchical validation of parameters:
 *
 * shipped default-parameters (in the inherited struct) have lowest precedence.
 * They will be overridden by a given configuration struct. However, even the
 * given configuration struct can be overridden by compile-time command line
 * parameters (e.g. -D POLICYMALLOC_DP_XMALLOCSIMD_PAGESIZE 1024)
 *
 * default-struct < template-struct < command-line parameter
 */
#ifndef POLICYMALLOC_DP_XMALLOCSIMD_PAGESIZE
#define POLICYMALLOC_DP_XMALLOCSIMD_PAGESIZE Properties::pagesize::value
#endif
      static const uint32 pagesize      = POLICYMALLOC_DP_XMALLOCSIMD_PAGESIZE;

      //all the properties must be unsigned integers > 0
      BOOST_STATIC_ASSERT(!std::numeric_limits<typename Properties::pagesize::type>::is_signed);
      BOOST_STATIC_ASSERT(pagesize > 0);

    public:
      static const uint32 _pagesize = pagesize;

      __device__ uint32 collect(uint32 bytes){

        can_use_coalescing = false;
        warpid = PolicyMalloc::warpid();
        myoffset = 0;
        threadcount = 0;

        //init with initial counter
        __shared__ uint32 warp_sizecounter[32];
        warp_sizecounter[warpid] = 16;

        //second half: make sure that all coalesced allocations can fit within one page
        //necessary for offset calculation
        bool coalescible = bytes > 0 && bytes < (pagesize / 32);
        uint32 threadcount = __popc(__ballot(coalescible));

        if (coalescible && threadcount > 1)
        {
          myoffset = atomicAdd(&warp_sizecounter[warpid], bytes);
          can_use_coalescing = true;
        }

        req_size = bytes;
        if (can_use_coalescing)
          req_size = (myoffset == 16) ? warp_sizecounter[warpid] : 0;

        return req_size;
      }


      __device__ void* distribute(void* allocatedMem){
        __shared__ char* warp_res[32];

        char* myalloc = (char*) allocatedMem;
        if (req_size && can_use_coalescing)
        {
          warp_res[warpid] = myalloc;
          if (myalloc != 0)
            *(uint32*)myalloc = threadcount;
        }
        __threadfence_block();

        void *myres = myalloc;
        if(can_use_coalescing)
        {
          if(warp_res[warpid] != 0)
            myres = warp_res[warpid] + myoffset;
          else
            myres = 0;
        }
        return myres;
      }

      __host__ static std::string classname(){
        std::stringstream ss;
        ss << "XMallocSIMD[" << pagesize << "]";
        return ss.str();
      }

  };

} //namespace DistributionPolicies

} //namespace PolicyMalloc
