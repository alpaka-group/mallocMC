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
#include "src/include/scatteralloc/ReservePoolPolicies.hpp"
#include "src/include/scatteralloc/AlignmentPolicies.hpp"
    


// configurate the CreationPolicy "Scatter" to modify the default behaviour
struct ScatterHeapConfig : PolicyMalloc::CreationPolicies::Scatter<>::HeapProperties{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashConfig : PolicyMalloc::CreationPolicies::Scatter<>::HashingProperties{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct XMallocConfig : PolicyMalloc::DistributionPolicies::XMallocSIMD<>::Properties {
  typedef ScatterHeapConfig::pagesize pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct ShrinkConfig : PolicyMalloc::AlignmentPolicies::Shrink<>::Properties {
  typedef boost::mpl::int_<16> dataAlignment;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef PolicyMalloc::PolicyAllocator< 
  PolicyMalloc::CreationPolicies::Scatter<ScatterHeapConfig, ScatterHashConfig>,
  PolicyMalloc::DistributionPolicies::XMallocSIMD<XMallocConfig>,
  PolicyMalloc::OOMPolicies::ReturnNull,
  PolicyMalloc::ReservePoolPolicies::SimpleCudaMalloc,
  PolicyMalloc::AlignmentPolicies::Shrink<ShrinkConfig>
  > ScatterAllocator;

// use "ScatterAllocator" as PolicyAllocator
POLICYMALLOC_SET_ALLOCATOR_TYPE(ScatterAllocator)
