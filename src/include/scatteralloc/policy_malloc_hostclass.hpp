#pragma once 

#include "policy_malloc_utils.hpp"
#include <boost/cstdint.hpp>
#include <boost/tuple/tuple.hpp>
#include <sstream>
#include <typeinfo>

namespace PolicyMalloc{

  template < 
     typename T_CreationPolicy, 
     typename T_DistributionPolicy, 
     typename T_OOMPolicy, 
     typename T_ReservePoolPolicy,
     typename T_AlignmentPolicy
       >
  struct PolicyAllocator : 
    public T_CreationPolicy, 
    public T_OOMPolicy, 
    public T_ReservePoolPolicy,
    public T_AlignmentPolicy
  {
    private:
      typedef boost::uint32_t uint32;
      typedef T_CreationPolicy CreationPolicy;
      typedef T_DistributionPolicy DistributionPolicy;
      typedef T_OOMPolicy OOMPolicy;
      typedef T_ReservePoolPolicy ReservePoolPolicy;
      typedef T_AlignmentPolicy AlignmentPolicy;
      void* pool;


    public:

      typedef PolicyAllocator<CreationPolicy,DistributionPolicy,OOMPolicy,ReservePoolPolicy,AlignmentPolicy> MyType;
      __device__ void* alloc(size_t bytes){
        DistributionPolicy distributionPolicy;

        bytes            = AlignmentPolicy::applyPadding(bytes);
        uint32 req_size  = distributionPolicy.collect(bytes);
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
        pool = ReservePoolPolicy::setMemPool(size);
        boost::tie(pool,size) = AlignmentPolicy::alignPool(pool,size);
        return CreationPolicy::initHeap(*this,pool,size);
      }

      __host__ void finalizeHeap(){
        CreationPolicy::finalizeHeap(*this);
        ReservePoolPolicy::resetMemPool(pool);
      }

      __host__ static std::string info(std::string linebreak = " "){
        std::stringstream ss;
        ss << "CreationPolicy:      " << CreationPolicy::classname()     << linebreak;
        ss << "DistributionPolicy:  " << DistributionPolicy::classname() << linebreak;
        ss << "OOMPolicy:           " << OOMPolicy::classname()          << linebreak;
        ss << "ReservePoolPolicy:   " << ReservePoolPolicy::classname()  << linebreak;
        ss << "AlignmentPolicy:     " << AlignmentPolicy::classname()    << linebreak;
        return ss.str();
      }

  };

} //namespace PolicyMalloc
