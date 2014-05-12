#pragma once 

#include "policy_malloc_utils.hpp"
#include "policy_malloc_constraints.hpp"
#include <boost/cstdint.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/mpl/bool.hpp>
#include <sstream>
#include <cassert>

#include <boost/mpl/assert.hpp>

namespace PolicyMalloc{

  /**
   * @brief Defines Traits for certain PolicyAllocators
   *
   * This trait class provides information about the capabilities of the
   * allocator.
   * Available traits:
   * bool providesAvailableSlots: declares if the allocator implements a useful
   * version of getAvailableSlots().
   */
  template <class T>
  class  Traits : public T{
    public:
    static const bool providesAvailableSlotsHost        = T::CreationPolicy::providesAvailableSlotsHost::value;
    static const bool providesAvailableSlotsAccelerator = T::CreationPolicy::providesAvailableSlotsAccelerator::value;
  };


  /**
   * @brief "HostClass" that combines all policies to a useful allocator
   *
   * This class implements the necessary glue-logic to form an actual allocator
   * from the provided policies. It implements the public interface and
   * executes some constraint checking based on an instance of the class
   * PolicyConstraints.
   *
   * @param T_CreationPolicy (template) The desired type of a CreationPolicy
   * @param T_DistributionPolicy (template) The desired type of a DistributionPolicy
   * @param T_OOMPolicy (template) The desired type of a OOMPolicy
   * @param T_ReservePoolPolicy (template) The desired type of a ReservePoolPolicy
   * @param T_AlignmentPolicy (template) The desired type of a AlignmentPolicy
   */
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
    public:
      typedef T_CreationPolicy CreationPolicy;
      typedef T_DistributionPolicy DistributionPolicy;
      typedef T_OOMPolicy OOMPolicy;
      typedef T_ReservePoolPolicy ReservePoolPolicy;
      typedef T_AlignmentPolicy AlignmentPolicy;

    private:
      typedef boost::uint32_t uint32;
      void* pool;

      //Instantiating the constraints checker will execute the check
      PolicyConstraints<CreationPolicy,DistributionPolicy,
        OOMPolicy,ReservePoolPolicy,AlignmentPolicy> c;

    public:

      typedef PolicyAllocator<CreationPolicy,DistributionPolicy,
              OOMPolicy,ReservePoolPolicy,AlignmentPolicy> MyType;

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
        CreationPolicy::finalizeHeap(*this,pool);
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

      // polymorphism over the availability of getAvailableSlotsHost
      __host__
      unsigned getAvailableSlotsHost(size_t slotSize){
        return getAvailSlotsHostPoly(slotSize, boost::mpl::bool_<CreationPolicy::providesAvailableSlotsHost::value>());
      }

      // polymorphism over the availability of getAvailableSlotsAccelerator
      __device__
      unsigned getAvailableSlotsAccelerator(size_t slotSize){
        return getAvailSlotsAcceleratorPoly(slotSize, boost::mpl::bool_<CreationPolicy::providesAvailableSlotsAccelerator::value>());
      }

    private:
      __host__
      unsigned getAvailSlotsHostPoly(size_t slotSize, boost::mpl::bool_<false>){
        assert(false);
        return 0;
      }

      __host__
      unsigned getAvailSlotsHostPoly(size_t slotSize, boost::mpl::bool_<true>){
        slotSize = AlignmentPolicy::applyPadding(slotSize);
        return CreationPolicy::getAvailableSlotsHost(slotSize,*this);
      }

      __device__
      unsigned getAvailSlotsAcceleratorPoly(size_t slotSize, boost::mpl::bool_<false>){
        assert(false);
        return 0;
      }

      __device__
      unsigned getAvailSlotsAcceleratorPoly(size_t slotSize, boost::mpl::bool_<true>){
        slotSize = AlignmentPolicy::applyPadding(slotSize);
        return CreationPolicy::getAvailableSlotsAccelerator(slotSize);
      }


  };

} //namespace PolicyMalloc
