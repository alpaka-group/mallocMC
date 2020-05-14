/*
  mallocMC: Memory Allocator for Many Core Architectures.
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

#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>
#include <stdio.h>

#include <cuda.h>


///////////////////////////////////////////////////////////////////////////////
// includes for mallocMC
///////////////////////////////////////////////////////////////////////////////
#include <mallocMC/mallocMC_hostclass.hpp>
#include <mallocMC/CreationPolicies.hpp>
#include <mallocMC/DistributionPolicies.hpp>
#include <mallocMC/OOMPolicies.hpp>
#include <mallocMC/ReservePoolPolicies.hpp>
#include <mallocMC/AlignmentPolicies.hpp>

///////////////////////////////////////////////////////////////////////////////
// Configuration for mallocMC
///////////////////////////////////////////////////////////////////////////////

// configurate the CreationPolicy "Scatter"
struct ScatterConfig{
    static constexpr auto pagesize = 4096;
    static constexpr auto accessblocks = 8;
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = 2;
    static constexpr auto resetfreedpages = false;
};

struct ScatterHashParams{
    static constexpr auto hashingK = 38183;
    static constexpr auto hashingDistMP = 17497;
    static constexpr auto hashingDistWP = 1;
    static constexpr auto hashingDistWPRel = 1;
};


// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig{
    static constexpr auto dataAlignment = 16;
};

// Define a new mMCator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
using ScatterAllocator = mallocMC::Allocator<
    mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
    mallocMC::DistributionPolicies::Noop,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
    mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>
>;

///////////////////////////////////////////////////////////////////////////////
// End of mallocMC configuration
///////////////////////////////////////////////////////////////////////////////


__device__ int* arA;


__global__ void exampleKernel(ScatterAllocator::AllocatorHandle mMC){
    unsigned x = 42;
    if(threadIdx.x==0)
        arA = (int*) mMC.malloc(sizeof(int) * 32);

    x = mMC.getAvailableSlots(1);
    __syncthreads();
    arA[threadIdx.x] = threadIdx.x;
    printf("tid: %d array: %d slots %d\n", threadIdx.x, arA[threadIdx.x],x);

    if(threadIdx.x == 0)
        mMC.free(arA);
}


int main()
{
    ScatterAllocator mMC(1U*1024U*1024U*1024U); //1GB for device-side malloc

    exampleKernel<<<1,32>>>( mMC );
    std::cout << "Slots from Host: " << mMC.getAvailableSlots(1) << std::endl;

    return 0;
}
