#pragma once

#include <boost/cstdint.hpp>
#include <boost/static_assert.hpp>
#include <stdio.h>
#include <iostream>
#include <boost/tuple/tuple.hpp>


#include "Shrink.hpp"

namespace PolicyMalloc{
namespace AlignmentPolicies{

namespace Shrink2NS{
    
  template<int PSIZE> struct __PointerEquivalent{ typedef unsigned int type;};
  template<>
  struct __PointerEquivalent<8>{ typedef unsigned long long int type; };

}// namespace ShrinkNS

  template<typename T_Dummy>
  class Shrink2{
    typedef boost::uint32_t uint32;
    typedef Shrink2<T_Dummy> MyType;
    typedef typename GetProperties<MyType>::dataAlignment DataAlignment;

    static const uint32 dataAlignment = DataAlignment::value;
    typedef Shrink2NS::__PointerEquivalent<sizeof(char*)>::type PointerEquivalent;

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

    static boost::tuple<void*,size_t> alignPool(void* memory, size_t memsize){
      PointerEquivalent alignmentstatus = ((PointerEquivalent)memory) & (dataAlignment -1);
      if(alignmentstatus != 0)
      {
        std::cout << "Heap Warning: memory to use not ";
        std::cout << dataAlignment << " byte aligned..."        << std::endl;
        std::cout << "Before:"                                  << std::endl;
        std::cout << "dataAlignment:   " << dataAlignment       << std::endl;
        std::cout << "Alignmentstatus: " << alignmentstatus     << std::endl;
        std::cout << "size_t memsize   " << memsize << " byte"  << std::endl;
        std::cout << "void *memory     " << memory              << std::endl;

        memory   = (void*)(((PointerEquivalent)memory) + dataAlignment - alignmentstatus);
        memsize -= (size_t)dataAlignment + (size_t)alignmentstatus;

        std::cout << "Was shrinked automatically to:"            << std::endl;
        std::cout << "size_t memsize   " << memsize << " byte"  << std::endl;
        std::cout << "void *memory     " << memory              << std::endl;
      }

      return boost::make_tuple(memory,memsize);
    }

    __device__ static uint32 alignAccess(uint32 bytes){
      return (bytes + dataAlignment - 1) & ~(dataAlignment-1);
    }

  };

} //namespace AlignmentPolicies
} //namespace PolicyMalloc
