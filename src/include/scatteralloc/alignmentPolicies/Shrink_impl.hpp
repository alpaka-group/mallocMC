#pragma once

#include <boost/cstdint.hpp>
#include <boost/static_assert.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <boost/tuple/tuple.hpp>

#include "Shrink.hpp"

namespace PolicyMalloc{
namespace AlignmentPolicies{

namespace Shrink2NS{
    
  template<int PSIZE> struct __PointerEquivalent{ typedef unsigned int type;};
  template<>
  struct __PointerEquivalent<8>{ typedef unsigned long long int type; };

}// namespace ShrinkNS

  template<typename T_Config>
  class Shrink{
    typedef boost::uint32_t uint32;
    typedef Shrink2NS::__PointerEquivalent<sizeof(char*)>::type PointerEquivalent;

    typedef T_Config Properties;

#ifdef POLICYMALLOC_AP_SHRINK_DATAALIGNMENT
    static const uint32 dataAlignment = POLICYMALLOC_AP_SHRINK_DATAALIGNMENT;
#else
    typedef typename Properties::dataAlignment DataAlignment;
    static const uint32 dataAlignment = DataAlignment::value;
    BOOST_STATIC_ASSERT(!std::numeric_limits<typename DataAlignment::type>::is_signed);
#endif //POLICYMALLOC_AP_SHRINK_DATAALIGNMENT

    BOOST_STATIC_ASSERT(dataAlignment > 0); 
    //dataAlignment must also be a power of 2!
    BOOST_STATIC_ASSERT(dataAlignment && !(dataAlignment & (dataAlignment-1)) ); 
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

    __host__ static std::string classname(){
      std::stringstream ss;
      ss << "Shrink[" << dataAlignment << "]";
      return ss.str();
    }

  };

} //namespace AlignmentPolicies
} //namespace PolicyMalloc
