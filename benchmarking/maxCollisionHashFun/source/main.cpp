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
#include <fstream>


using mallocMC::CreationPolicies::FlatterScatter;

using Dim = alpaka::DimInt<1>;
using Idx = std::size_t;

// Define the device accelerator
using Acc = alpaka::ExampleDefaultAcc<Dim, Idx>;

constexpr uint32_t const blocksize = 16U * 1024U * 1024U;
constexpr uint32_t const pagesize = 4U * 1024U;
constexpr uint32_t const wasteFactor = 1U;
constexpr uint32_t const allocSize = 128U;
constexpr uint32_t const noOfMeasurements = 1000U;
constexpr uint32_t const maxNoOfTHreads = 100U;
constexpr uint32_t const noOfChunks = 31;
constexpr uint32_t const requiredPages = 1;
constexpr uint32_t const maskSize = 32;


// This happens to also work for the original Scatter algorithm, so we only define one.
struct FlatterScatterHeapConfig : FlatterScatter<>::Properties::HeapConfig
{
    static constexpr auto accessblocksize = blocksize;
    static constexpr auto pagesize = ::pagesize;
    static constexpr auto heapsize = 2U * 1024U * 1024U * 1024U;
    // Only used by original Scatter (but it doesn't hurt FlatterScatter to keep):
    static constexpr auto regionsize = 16;
    static constexpr auto wastefactor = wasteFactor;
};

struct FlatterScatterHashConfig : FlatterScatter<>::Properties::HashConfig
{
    static constexpr uint32_t blockStride = 0;

    template<uint32_t T_pageSize, typename TAcc>
    ALPAKA_FN_INLINE ALPAKA_FN_ACC static auto hash(TAcc const& acc, uint32_t const numBytes) -> uint32_t
    {
        return 0;
    }
};

struct XMallocConfig
{
    static constexpr auto pagesize = FlatterScatterHeapConfig::pagesize;
};

struct ShrinkConfig
{
    static constexpr auto dataAlignment = 16;
};


ALPAKA_STATIC_ACC_MEM_GLOBAL int** arA;


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
    auto const heapSize = 2U * 1024U * 1024U * 1024U;
    std::cerr << "done\n";
    std::cout << Allocator::info("\n") << '\n';

    // create arrays of arrays on the device
    {
        int globalAtomicsSum = 0;

        auto allocArray
            = [] ALPAKA_FN_ACC(Acc const& acc, int x, Allocator::AllocatorHandle allocHandle)
        {
            arA<Acc> = static_cast<int**>(allocHandle.malloc(acc, sizeof(int*) * x));
        };

        auto freeArray
            = [] ALPAKA_FN_ACC(Acc const& acc, Allocator::AllocatorHandle allocHandle)
        {
            allocHandle.free(acc, arA<Acc>);
        };

        auto allocMemory
            = [] ALPAKA_FN_ACC(Acc const& acc, Allocator::AllocatorHandle allocHandle)
        {
            auto const id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
            // printf("alloc ");
            arA<Acc>[id] = static_cast<int*>(allocHandle.malloc(acc, allocSize));
        };

        auto freeMemory
            = [] ALPAKA_FN_ACC(Acc const& acc, Allocator::AllocatorHandle allocHandle)
        {
            auto const id = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0]; 
            // printf("free ");  
            allocHandle.free(acc, arA<Acc>[id]);
        };

        Allocator scatterAlloc(dev, queue, heapSize); // 2GB for device-side malloc
        std::array<uint32_t, maxNoOfTHreads> arrayOfThreads{};

        uint32_t noOfThreads = 1;
        for(auto& thread : arrayOfThreads)
        {
            thread = noOfThreads++;
        }

        std::ofstream csv("results.csv");
        csv << "thread_count";
        for(int i = 1; i <= noOfMeasurements; ++i)
            csv << ",measurement" << i;
        csv << ",mean";
        csv << ",noOfAtmoics\n";

        Allocator::CreationPolicy::template 
        initHeap<Acc>(dev, queue, scatterAlloc.getAllocatorHandle().devAllocator, scatterAlloc.getHeapLocations()[0].p ,heapSize);

        auto const workDivSingle = alpaka::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}};

        for(auto const& thread: arrayOfThreads)
        {
            auto const workDiv = alpaka::WorkDivMembers<Dim, Idx>{Idx{thread}, Idx{1}, Idx{1}};
            std::cout << "Number Of Threads: " << thread << '\n';
            std::array<std::chrono::duration<double>, noOfMeasurements> measurements {};

            for(auto& measurement : measurements)
            {
                alpaka::enqueue(
                    queue,
                    alpaka::createTaskKernel<Acc>(
                        workDivSingle,
                        allocArray,
                        thread,
                        scatterAlloc.getAllocatorHandle()
                    )
                );

                alpaka::enqueue(
                    queue,
                    alpaka::createTaskKernel<Acc>(
                        workDiv,
                        allocMemory,
                        scatterAlloc.getAllocatorHandle()
                    )
                );

                std::cout << "New measurement\n";
                
                const auto start = std::chrono::high_resolution_clock::now();

                alpaka::enqueue(
                    queue,
                    alpaka::createTaskKernel<Acc>(
                        workDiv,
                        freeMemory,
                        scatterAlloc.getAllocatorHandle()
                    )
                );

                const auto end = std::chrono::high_resolution_clock::now();
                measurement = end - start;
                
                alpaka::enqueue(
                    queue,
                    alpaka::createTaskKernel<Acc>(
                        workDivSingle,
                        freeArray,
                        scatterAlloc.getAllocatorHandle()
                    )
                );

            }
            std::copy(measurements.cbegin(), measurements.cend(), std::ostream_iterator<std::chrono::duration<double>>(std::cout, ", "));
            std::cout << "\n";
            auto mean = std::reduce(measurements.cbegin(), measurements.cend()) / noOfMeasurements;
            std::cout << "mean: " << mean << "\n";

            int localAtomicsSum = 0;

            for (int i = 0; i < thread; i++)
            {
                localAtomicsSum += 4 + ((i % noOfChunks) > 0 ? 0 : 3);
            }
            
            localAtomicsSum *= noOfMeasurements;
            globalAtomicsSum += localAtomicsSum;

            csv << thread;
            for(auto const& m : measurements)
                csv << "," << m.count();
            csv << "," << mean.count();
            csv << "," << localAtomicsSum/noOfMeasurements << "\n";
        }

        std::cout << "Total number of atmoic operations: " << globalAtomicsSum << '\n';

    }

    return 0;
}

auto main(int /*argc*/, char* /*argv*/[]) -> int
{
    maxCollisionHashFun<FlatterScatter<FlatterScatterHeapConfig, FlatterScatterHashConfig>, mallocMC::ReservePoolPolicies::AlpakaBuf<Acc>>();

    return 0;
}
