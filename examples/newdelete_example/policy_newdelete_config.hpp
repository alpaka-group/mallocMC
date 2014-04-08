#pragma once

#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>

// basic files for PolicyMalloc
#include "src/include/scatteralloc/policy_malloc_overwrites.hpp"
#include "src/include/scatteralloc/policy_malloc_hostclass.hpp"

// Load all available policies for PolicyMalloc
#include "src/include/scatteralloc/CreationPolicies.hpp"
#include "src/include/scatteralloc/DistributionPolicies.hpp"
#include "src/include/scatteralloc/OOMPolicies.hpp"
#include "src/include/scatteralloc/GetHeapPolicies.hpp"
#include "src/include/scatteralloc/AlignmentPolicies.hpp"
    
// Define a new allocator and call it ScatterAllocator
typedef PolicyMalloc::PolicyAllocator< 
  PolicyMalloc::CreationPolicies::Scatter,
  PolicyMalloc::DistributionPolicies::XMallocSIMD,
  PolicyMalloc::OOMPolicies::ReturnNull,
  PolicyMalloc::GetHeapPolicies::SimpleCudaMalloc,
  PolicyMalloc::AlignmentPolicies::Shrink
  > ScatterAllocator;


// configurate the CreationPolicy "Scatter"
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

// configure the DistributionPolicy "XMallocSIMD"
template<>
struct PolicyMalloc::GetProperties<PolicyMalloc::DistributionPolicies::XMallocSIMD>{
  typedef GetProperties<CreationPolicies::Scatter>::pagesize pagesize;
};

// configure the AlignmentPolicy "Shrink"
template<>
struct PolicyMalloc::GetProperties<PolicyMalloc::AlignmentPolicies::Shrink>{
  typedef boost::mpl::int_<16> dataAlignment;
};

// use "ScatterAllocator" as PolicyAllocator
SET_ACCELERATOR_MEMORY_ALLOCATOR_TYPE(ScatterAllocator)

// replace all standard malloc()-calls on the device by PolicyAllocator calls
// This will not work with the CreationPolicy "OldMalloc"!
POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE()

