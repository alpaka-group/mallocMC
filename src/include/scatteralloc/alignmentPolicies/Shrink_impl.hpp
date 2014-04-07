#pragma once

#include <boost/cstdint.hpp>
#include <boost/static_assert.hpp>
#include <stdio.h>

#include "Shrink.hpp"

namespace PolicyMalloc{
namespace AlignmentPolicies{

namespace Shrink2NS{
    
  typedef boost::uint32_t uint32;
  template<int PSIZE> struct __PointerEquivalent{ typedef unsigned int type;};
  template<>
  struct __PointerEquivalent<8>{ typedef unsigned long long int type; };

  typedef __PointerEquivalent<sizeof(char*)>::type PointerEquivalent;

  __global__ void alignPoolKernel(void* memory, uint32 dataAlignment){
    PointerEquivalent alignmentstatus = ((PointerEquivalent)memory) & (dataAlignment -1);
    if(alignmentstatus != 0)
    {
      memory =(void*)(((PointerEquivalent)memory) + dataAlignment - alignmentstatus);
      printf("Heap Warning: memory to use not 16 byte aligned...\n");
    }
  }
}// namespace ShrinkNS

  template<typename T_Dummy>
  class Shrink2{
    typedef boost::uint32_t uint32;
    typedef Shrink2<T_Dummy> MyType;
    typedef typename GetProperties<MyType>::dataAlignment DataAlignment;

    static const uint32 dataAlignment = DataAlignment::value;

#ifndef BOOST_NOINLINE
#define BOOST_NOINLINE='__attribute__ ((noinline)'
#define BOOST_NOINLINE_WAS_JUSTDEFINED
#endif
    BOOST_STATIC_ASSERT(!std::numeric_limits<typename DataAlignment::type>::is_signed);
    BOOST_STATIC_ASSERT(dataAlignment > 0); 
    //dataAlignment must also be a power of 2!
    BOOST_STATIC_ASSERT(dataAlignment && !(dataAlignment & (dataAlignment-1)) ); 
#ifdef BOOST_NOINLINE_WAS_JUSTDEFINED
#undef BOOST_NOINLINE_WAS_JUSTDEFINED
#undef BOOST_NOINLINE
#endif

    public:

    static void* alignPool(void* memory){
      Shrink2NS::alignPoolKernel<<<1,1>>>(memory,dataAlignment);
      //TODO:maybe also take care of the memory-size bug
      return memory;
    }

    __device__ static uint32 alignAccess(uint32 bytes){
      return (bytes + dataAlignment - 1) & ~(dataAlignment-1);
    }

  };

} //namespace AlignmentPolicies
} //namespace PolicyMalloc
