/*
  ScatterAlloc: Massively Parallel Dynamic Memory Allocation for the GPU.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

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
  PolicyMalloc::ReservePoolPolicies::SimpleCudaMalloc,
  PolicyMalloc::AlignmentPolicies::Shrink<AlignmentConfig>
  > ScatterAllocator;
