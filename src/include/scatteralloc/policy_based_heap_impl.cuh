#pragma once

template<typename T>
struct GetProperties{};

#include "policy_based_heap.cuh"
#include "get_heap_simpleMalloc.cuh" /*GetHeapPolicy: GetHeapSimpleMalloc*/
#include "xmalloc_like_distribution.cuh" /*AllocationPolicy: XMallocDistribution*/
#include "null_on_oom_policy.cuh"    /*OOMPolicy: NullOnOOM*/
#include "scatterd_heap_policy.cuh"  /*CreationPolicy: ScatteredHeap */
#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>


// global type
typedef GPUTools::PolicyAllocator< 
  GPUTools::ScatteredHeap<>, 
  GPUTools::XMallocDistribution<>, 
  NullOnOOM, 
  GPUTools::GetHeapSimpleMalloc 
  > ScatterAllocator;


//parameters for the Heap
template<>
struct GetProperties<GPUTools::ScatteredHeap<> >{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
  //struct value{
  //  enum{
  //    pagesize        = 4096,
  //    accessblocks    = 8,
  //    regionsize      = 16,
  //    wastefactor     = 2,
  //    resetfreedpages = false
  //  };
  //};
};


//parameters for the Allocation Policy
template<>
struct GetProperties<GPUTools::XMallocDistribution<> >{
  struct value{
    enum { pagesize = 4096 };
  };
};


//typedef OtherAllocator PolClass;
typedef  ScatterAllocator PolClass;


// global Heap Object
__device__ PolClass polObject;

// global initHeap
__host__ void* initHeap(
    size_t heapsize = 8U*1024U*1024U,
    PolClass p = polObject
    ){
  return PolClass::initHeap(p,heapsize);
};

// global destroyHeap
__host__ void destroyHeap(PolClass p = polObject){
  PolClass::destroyHeap(p);
};

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 200
// global overwrite malloc/free
__device__ void* malloc(size_t t) __THROW
{
  return polObject.alloc(t);
};
__device__ void  free(void* p) __THROW
{
  polObject.dealloc(p);
};
#endif
#endif

//TODO: globally overwrite new, new[], delete, delete[], placment new, placement new[]
