#pragma once 

#include "policy_malloc_utils.hpp"
#include <boost/cstdint.hpp>
#include <boost/tuple/tuple.hpp>

namespace PolicyMalloc{

  template <typename T>
  struct GetProperties;

  template < 
     typename T_CreationPolicy, 
     typename T_DistributionPolicy, 
     typename T_OOMPolicy, 
     typename T_GetHeapPolicy,
     typename T_AlignmentPolicy
       >
  struct PolicyAllocator : 
    public T_CreationPolicy, 
    public T_OOMPolicy, 
    public T_GetHeapPolicy,
    public T_AlignmentPolicy
  {
    private:
      typedef boost::uint32_t uint32;
      typedef T_CreationPolicy CreationPolicy;
      typedef T_DistributionPolicy DistributionPolicy;
      typedef T_OOMPolicy OOMPolicy;
      typedef T_GetHeapPolicy GetHeapPolicy;
      typedef T_AlignmentPolicy AlignmentPolicy;
      void* pool;

    public:

      typedef PolicyAllocator<CreationPolicy,DistributionPolicy,OOMPolicy,GetHeapPolicy,AlignmentPolicy> MyType;
      __device__ void* alloc(size_t bytes){
        DistributionPolicy distributionPolicy;

        bytes            = AlignmentPolicy::alignAccess(bytes);
        uint32 req_size  = distributionPolicy.collect(bytes);
        req_size         = AlignmentPolicy::alignAccess(req_size); //TODO check if this call is necessary
        void* memBlock   = CreationPolicy::create(req_size);
        const bool oom   = CreationPolicy::isOOM(memBlock);
        if(oom) memBlock = OOMPolicy::handleOOM(memBlock);
        void* myPart     = distributionPolicy.distribute(memBlock);

        return myPart;
        // if(blockIdx.x==0 && threadIdx.x==0){
        //     printf("warp %d trying to allocate %d bytes. myalloc: %p (oom %d)\n",GPUTools::warpid(),req_size,myalloc,oom);
        // }
      }

      __device__ void dealloc(void* p){
        CreationPolicy::destroy(p);
      }

      __host__ void* initHeap(size_t size){
        pool = GetHeapPolicy::setMemPool(size);
        boost::tie(pool,size) = AlignmentPolicy::alignPool(pool,size);
        return CreationPolicy::initHeap(*this,pool,size);
      }

      __host__ void finalizeHeap(){
        CreationPolicy::finalizeHeap(*this);
        GetHeapPolicy::resetMemPool(pool);
      }

  };

} //namespace PolicyMalloc
