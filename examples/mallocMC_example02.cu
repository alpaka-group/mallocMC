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

#include <cassert>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// includes for mallocMC
///////////////////////////////////////////////////////////////////////////////
// basic files for mallocMC
#include <mallocMC/mallocMC_hostclass.hpp>

// Load all available policies for mallocMC
#include <mallocMC/AlignmentPolicies.hpp>
#include <mallocMC/CreationPolicies.hpp>
#include <mallocMC/DistributionPolicies.hpp>
#include <mallocMC/OOMPolicies.hpp>
#include <mallocMC/ReservePoolPolicies.hpp>

///////////////////////////////////////////////////////////////////////////////
// Configuration for mallocMC
///////////////////////////////////////////////////////////////////////////////

// configurate the CreationPolicy "Scatter"
struct ScatterConfig
{
    static constexpr auto pagesize = 4096;
    static constexpr auto accessblocks = 8;
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = 2;
    static constexpr auto resetfreedpages = false;
};

struct ScatterHashParams
{
    static constexpr auto hashingK = 38183;
    static constexpr auto hashingDistMP = 17497;
    static constexpr auto hashingDistWP = 1;
    static constexpr auto hashingDistWPRel = 1;
};

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig
{
    static constexpr auto pagesize = ScatterConfig::pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig
{
    static constexpr auto dataAlignment = 16;
};

// Define a new mMCator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
using ScatterAllocator = mallocMC::Allocator<
    mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
    mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
    mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

///////////////////////////////////////////////////////////////////////////////
// End of mallocMC configuration
///////////////////////////////////////////////////////////////////////////////

void run();

auto main() -> int
{
    int computeCapabilityMajor = 0;
    cudaDeviceGetAttribute(
        &computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, 0);
    int computeCapabilityMinor = 0;
    cudaDeviceGetAttribute(
        &computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, 0);

    if(computeCapabilityMajor < int(2))
    {
        std::cerr << "Error: Compute Capability >= 2.0 required. (is ";
        std::cerr << computeCapabilityMajor << "." << computeCapabilityMinor
                  << ")" << std::endl;
        return 1;
    }

    cudaSetDevice(0);
    run();
    cudaDeviceReset();

    return 0;
}

__device__ int ** arA;
__device__ int ** arB;
__device__ int ** arC;

__global__ void
createArrayPointers(int x, int y, ScatterAllocator::AllocatorHandle mMC)
{
    arA = (int **)mMC.malloc(sizeof(int *) * x * y);
    arB = (int **)mMC.malloc(sizeof(int *) * x * y);
    arC = (int **)mMC.malloc(sizeof(int *) * x * y);
}

__global__ void
fillArrays(int length, int * d, ScatterAllocator::AllocatorHandle mMC)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    arA[id] = (int *)mMC.malloc(sizeof(int) * length);
    arB[id] = (int *)mMC.malloc(sizeof(int) * length);
    arC[id] = (int *)mMC.malloc(sizeof(int) * length);

    for(int i = 0; i < length; ++i)
    {
        arA[id][i] = id * length + i;
        arB[id][i] = id * length + i;
    }
}

__global__ void addArrays(int length, int * d)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    d[id] = 0;
    for(int i = 0; i < length; ++i)
    {
        arC[id][i] = arA[id][i] + arB[id][i];
        d[id] += arC[id][i];
    }
}

__global__ void freeArrays(ScatterAllocator::AllocatorHandle mMC)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    mMC.free(arA[id]);
    mMC.free(arB[id]);
    mMC.free(arC[id]);
}

__global__ void freeArrayPointers(ScatterAllocator::AllocatorHandle mMC)
{
    mMC.free(arA);
    mMC.free(arB);
    mMC.free(arC);
}

void run()
{
    size_t block = 32;
    size_t grid = 32;
    int length = 100;
    assert((unsigned)length <= block * grid); // necessary for used algorithm

    // init the heap
    std::cerr << "initHeap...";
    ScatterAllocator mMC(
        1U * 1024U * 1024U * 1024U); // 1GB for device-side malloc
    std::cerr << "done" << std::endl;

    // device-side pointers
    int * d;
    cudaMalloc((void **)&d, sizeof(int) * block * grid);

    // host-side pointers
    std::vector<int> array_sums(block * grid, 0);

    // create arrays of arrays on the device
    createArrayPointers<<<1, 1>>>(grid, block, mMC);

    // fill 2 of them all with ascending values
    fillArrays<<<grid, block>>>(length, d, mMC);

    // add the 2 arrays (vector addition within each thread)
    // and do a thread-wise reduce to d
    addArrays<<<grid, block>>>(length, d);

    cudaMemcpy(
        &array_sums[0], d, sizeof(int) * block * grid, cudaMemcpyDeviceToHost);

    int sum = std::accumulate(array_sums.begin(), array_sums.end(), 0);
    std::cout << "The sum of the arrays on GPU is " << sum << std::endl;

    int n = block * grid * length;
    int gaussian = n * (n - 1);
    std::cout << "The gaussian sum as comparison: " << gaussian << std::endl;

    // checking the free memory of the allocator
    if(mallocMC::Traits<ScatterAllocator>::providesAvailableSlots)
    {
        std::cout << "there are ";
        std::cout << mMC.getAvailableSlots(1024U * 1024U);
        std::cout << " Slots of size 1MB available" << std::endl;
    }

    freeArrays<<<grid, block>>>(mMC);
    freeArrayPointers<<<1, 1>>>(mMC);
    cudaFree(d);
}
