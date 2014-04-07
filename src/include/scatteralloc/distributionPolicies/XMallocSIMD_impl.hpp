#pragma once

#include <boost/cstdint.hpp>
#include <boost/static_assert.hpp>

#include "../policy_malloc_utils.hpp"
#include "XMallocSIMD.hpp"

namespace PolicyMalloc{
namespace DistributionPolicies{

  template<class T_Dummy>
  class XMallocSIMD2
  {
    private:

      typedef boost::uint32_t uint32;
      bool can_use_coalescing;
      uint32 warpid;
      uint32 myoffset;
      uint32 threadcount;
      uint32 req_size;
      typedef XMallocSIMD2<T_Dummy> MyType;
      typedef GetProperties<MyType> Properties;
      static const uint32 pagesize      = Properties::pagesize::value;

#ifndef BOOST_NOINLINE
#define BOOST_NOINLINE='__attribute__ ((noinline)'
#define BOOST_NOINLINE_WAS_JUSTDEFINED
#endif
      //all the properties must be unsigned integers > 0

      BOOST_STATIC_ASSERT(!std::numeric_limits<typename Properties::pagesize::type>::is_signed);
      BOOST_STATIC_ASSERT(pagesize > 0);

#ifdef BOOST_NOINLINE_WAS_JUSTDEFINED
#undef BOOST_NOINLINE_WAS_JUSTDEFINED
#undef BOOST_NOINLINE
#endif

    public:

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

  };

} //namespace DistributionPolicies

} //namespace PolicyMalloc
