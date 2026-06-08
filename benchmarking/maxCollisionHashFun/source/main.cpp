/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 - 2024 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de
              Julian Lenz - j.lenz ( at ) hzdr.de

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

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>

#include <mallocMC/mallocMC.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <chrono>


using mallocMC::CreationPolicies::FlatterScatter;

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

constexpr uint32_t const blocksize = 16U /** 1024U8*/ * 1024U;
constexpr uint32_t const pagesize = 4U * 1024U;
constexpr uint32_t const wasteFactor = 1U;
constexpr uint32_t const allocSize = 128U;
constexpr uint32_t const noOfMeasurements = 10;


// This happens to also work for the original Scatter algorithm, so we only define one.
struct FlatterScatterHeapConfig : FlatterScatter<>::Properties::HeapConfig
{
    static constexpr auto accessblocksize = blocksize;
    static constexpr auto pagesize = ::pagesize;
    static constexpr auto heapsize = /*2U * 1024U **/ 1024U * 1024U;
    // Only used by original Scatter (but it doesn't hurt FlatterScatter to keep):
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = wasteFactor;
};

struct XMallocConfig
{
    static constexpr auto pagesize = FlatterScatterHeapConfig::pagesize;
};

struct ShrinkConfig
{
    static constexpr auto dataAlignment = 16;
};

template<
    typename T_CreationPolicy,
    typename T_ReservePoolPolicy,
    typename T_AlignmentPolicy = mallocMC::AlignmentPolicies::Shrink<ShrinkConfig>>
auto maxCollisionHashFun() -> int
{
    using Allocator = mallocMC::Allocator<
        alpaka::AccToTag<Acc>,
        T_CreationPolicy,
        mallocMC::DistributionPolicies::Noop,
        mallocMC::OOMPolicies::ReturnNull,
        T_ReservePoolPolicy,
        T_AlignmentPolicy>;

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{dev};

    // init the heap
    std::cerr << "initHeap...";
    auto const heapSize = /*2U * 1024U **/ 1024U * 1024U;
    std::cerr << "done\n";
    std::cout << Allocator::info("\n") << '\n';

    // create arrays of arrays on the device
    {
        auto allocMemory
            = [] ALPAKA_FN_ACC(Acc const& acc, Allocator::AllocatorHandle allocHandle)
        {
            allocHandle.malloc(acc, allocSize);
        };

        Allocator scatterAlloc(dev, queue, heapSize); // 2GB for device-side malloc
        std::array<uint32_t, 12> noOfThreads{1,2,3,4,5,6,7,8,9,10,20, 32};

        for(auto const thread: noOfThreads)
        {
        auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{thread}, Idx{1}, Idx{1}};
        std::cout << "Number Of Threards: " << thread << '\n';
        std::array<std::chrono::duration<double>, noOfMeasurements> measurements {};

        for(auto& measurement : measurements)
        {
            Allocator::CreationPolicy::template 
            initHeap<Acc>(dev, queue, scatterAlloc.getAllocatorHandle().devAllocator, scatterAlloc.getHeapLocations()[0].p ,heapSize);
            const auto start = std::chrono::high_resolution_clock::now();

            alpaka::enqueue(
                queue,
                alpaka::createTaskKernel<Acc>(
                    workDiv,
                    allocMemory,
                    scatterAlloc.getAllocatorHandle()
                )
            );

            const auto end = std::chrono::high_resolution_clock::now();
            measurement = end - start;
        }
        std::copy(measurements.cbegin(), measurements.cend(), std::ostream_iterator<std::chrono::duration<double>>(std::cout, ", "));
        std::cout << "\n";
        std::cout << "mean: " << std::reduce(measurements.cbegin(), measurements.cend()) / noOfMeasurements << "\n";
    }
    }

    return 0;
}

auto main(int /*argc*/, char* /*argv*/[]) -> int
{
    maxCollisionHashFun<FlatterScatter<FlatterScatterHeapConfig>, mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>>();

    return 0;
}
