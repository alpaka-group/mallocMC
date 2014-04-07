#pragma once

#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>

#include "policy_malloc_overwrites.hpp"
#include "policy_malloc_hostclass.hpp"

#include "CreationPolicies.hpp"
#include "DistributionPolicies.hpp"
#include "OOMPolicies.hpp"
#include "GetHeapPolicies.hpp"
#include "AlignmentPolicies.hpp"
    
    

typedef PolicyMalloc::PolicyAllocator< 
  PolicyMalloc::CreationPolicies::Scatter,
  PolicyMalloc::DistributionPolicies::XMallocSIMD,
  PolicyMalloc::OOMPolicies::ReturnNull,
  PolicyMalloc::GetHeapPolicies::SimpleCudaMalloc,
  PolicyMalloc::AlignmentPolicies::Shrink
  > ScatterAllocator;

typedef PolicyMalloc::PolicyAllocator< 
  PolicyMalloc::CreationPolicies::OldMalloc,
  PolicyMalloc::DistributionPolicies::Noop,
  PolicyMalloc::OOMPolicies::ReturnNull,
  PolicyMalloc::GetHeapPolicies::CudaSetLimits,
  PolicyMalloc::AlignmentPolicies::Shrink
  > OldAllocator;

template<>
struct PolicyMalloc::GetProperties<PolicyMalloc::CreationPolicies::Scatter>{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;

    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

template<>
struct PolicyMalloc::GetProperties<PolicyMalloc::DistributionPolicies::XMallocSIMD>{
  typedef GetProperties<CreationPolicies::Scatter>::pagesize pagesize;
};

template<>
struct PolicyMalloc::GetProperties<PolicyMalloc::AlignmentPolicies::Shrink>{
  typedef boost::mpl::int_<16> dataAlignment;
};

SET_ACCELERATOR_MEMORY_ALLOCATOR_TYPE(ScatterAllocator)

//SET_ACCELERATOR_MEMORY_ALLOCATOR_TYPE(OldAllocator)

POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE()
