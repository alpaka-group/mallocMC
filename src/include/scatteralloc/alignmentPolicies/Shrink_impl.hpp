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
  template<typename T_Config>
  class Shrink{
    public:
    typedef T_Config Properties;

    private:
    typedef boost::uint32_t uint32;
    typedef Shrink2NS::__PointerEquivalent<sizeof(char*)>::type PointerEquivalent;

/** Allow for a hierarchical validation of parameters:
 *
 * shipped default-parameters (in the inherited struct) have lowest precedence.
 * They will be overridden by a given configuration struct. However, even the
 * given configuration struct can be overridden by compile-time command line
 * parameters (e.g. -D POLICYMALLOC_AP_SHRINK_DATAALIGNMENT 128)
 *
 * default-struct < template-struct < command-line parameter
 */
#ifndef POLICYMALLOC_AP_SHRINK_DATAALIGNMENT
#define POLICYMALLOC_AP_SHRINK_DATAALIGNMENT Properties::dataAlignment::value
#endif
    static const uint32 dataAlignment = POLICYMALLOC_AP_SHRINK_DATAALIGNMENT;

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

        std::cout << "Was shrunk automatically to: "            << std::endl;
        std::cout << "size_t memsize   " << memsize << " byte"  << std::endl;
        std::cout << "void *memory     " << memory              << std::endl;
      }

      return boost::make_tuple(memory,memsize);
    }

    __host__ __device__ static uint32 applyPadding(uint32 bytes){
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
