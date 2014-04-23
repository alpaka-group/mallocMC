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
    


// configurate the CreationPolicy "Scatter"
struct ScatterConfig{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashParams{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig{
  typedef ScatterConfig::pagesize pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig{
  typedef boost::mpl::int_<16> dataAlignment;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef PolicyMalloc::PolicyAllocator< 
  PolicyMalloc::CreationPolicies::Scatter<ScatterConfig,ScatterHashParams>,
  PolicyMalloc::DistributionPolicies::XMallocSIMD<DistributionConfig>,
  PolicyMalloc::OOMPolicies::ReturnNull,
  PolicyMalloc::GetHeapPolicies::SimpleCudaMalloc,
  PolicyMalloc::AlignmentPolicies::Shrink<AlignmentConfig>
  > ScatterAllocator;

// use "ScatterAllocator" as PolicyAllocator
POLICYMALLOC_SET_ALLOCATOR_TYPE(ScatterAllocator)
