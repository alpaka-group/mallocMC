/*
  mallocMC: Memory Allocator for Many Core Architectures.

  Copyright 2020 Helmholtz-Zentrum Dresden - Rossendorf,
                 CERN

  Author(s):  Bernhard Manfred Gruber

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
#include <mallocMC/AlignmentPolicies.hpp>
#include <mallocMC/CreationPolicies.hpp>
#include <mallocMC/DistributionPolicies.hpp>
#include <mallocMC/OOMPolicies.hpp>
#include <mallocMC/ReservePoolPolicies.hpp>
#include <mallocMC/mallocMC_hostclass.hpp>

using Idx = std::size_t;

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

struct DistributionConfig
{
    static constexpr auto pagesize = ScatterConfig::pagesize;
};

struct AlignmentConfig
{
    static constexpr auto dataAlignment = 16;
};

ALPAKA_STATIC_ACC_MEM_GLOBAL int ** deviceArray;

void test1D()
{
    using Dim = alpaka::dim::DimInt<1>;
    // using Acc = alpaka::acc::AccCpuThreads<Dim, Idx>;
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Idx>;

    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::Noop,
        // mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
        mallocMC::OOMPolicies::ReturnNull,
        // mallocMC::ReservePoolPolicies::SimpleMalloc,
        mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

    const auto dev
        = alpaka::pltf::getDevByIdx<alpaka::pltf::Pltf<alpaka::dev::Dev<Acc>>>(
            0);
    auto queue = alpaka::queue::Queue<Acc, alpaka::queue::Blocking>{dev};

    constexpr auto N = 1024;

    ScatterAllocator scatterAlloc(dev, queue, 1024U * 1024U); // 1 MiB

    // make 1 allocation from 1 thread for N pointers
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                int N,
                ScatterAllocator::AllocatorHandle allocHandle) {
                deviceArray
                    = (int **)allocHandle.malloc(acc, sizeof(int *) * N);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // make N allocations from N threads for ints
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{N}, Idx{1}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                ScatterAllocator::AllocatorHandle allocHandle) {
                const auto i
                    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(
                        acc)[0];
                deviceArray[i] = (int *)allocHandle.malloc(acc, sizeof(int));
            },
            scatterAlloc.getAllocatorHandle()));

    const auto slots = scatterAlloc.getAvailableSlots(dev, queue, sizeof(int));
    const auto heapInfo = scatterAlloc.getHeapLocations().at(0);
    std::cout << "slots: " << slots << " heap size: " << heapInfo.size << '\n';

    // free N allocations from N threads for ints
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{N}, Idx{1}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                ScatterAllocator::AllocatorHandle allocHandle) {
                const auto i
                    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(
                        acc)[0];
                allocHandle.free(acc, deviceArray[i]);
            },
            scatterAlloc.getAllocatorHandle()));

    // free 1 allocation from 1 thread for N pointers
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                ScatterAllocator::AllocatorHandle allocHandle) {
                allocHandle.free(acc, deviceArray);
            },
            scatterAlloc.getAllocatorHandle()));
}

void test2D()
{
    using Dim = alpaka::dim::DimInt<2>;
    // using Acc = alpaka::acc::AccCpuThreads<Dim, Idx>;
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Idx>;

    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::Noop,
        // mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
        mallocMC::OOMPolicies::ReturnNull,
        // mallocMC::ReservePoolPolicies::SimpleMalloc,
        mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

    const auto dev
        = alpaka::pltf::getDevByIdx<alpaka::pltf::Pltf<alpaka::dev::Dev<Acc>>>(
            0);
    auto queue = alpaka::queue::Queue<Acc, alpaka::queue::Blocking>{dev};

    constexpr auto N = 32;

    ScatterAllocator scatterAlloc(dev, queue, 1024U * 1024U); // 1 MiB

    // make 1 allocation from 1 thread for N*N pointers
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                int N,
                ScatterAllocator::AllocatorHandle allocHandle) {
                deviceArray
                    = (int **)allocHandle.malloc(acc, sizeof(int *) * N * N);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // make N*N allocations from N*N threads for ints
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{N}, Idx{N}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                int N,
                ScatterAllocator::AllocatorHandle allocHandle) {
                const auto idx
                    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                deviceArray[idx[0] * N + idx[1]]
                    = (int *)allocHandle.malloc(acc, sizeof(int));
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    const auto slots = scatterAlloc.getAvailableSlots(dev, queue, sizeof(int));
    const auto heapInfo = scatterAlloc.getHeapLocations().at(0);
    std::cout << "slots: " << slots << " heap size: " << heapInfo.size << '\n';

    // free N*N allocations from N*N threads for ints
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{N}, Idx{N}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                int N,
                ScatterAllocator::AllocatorHandle allocHandle) {
                const auto idx
                    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                allocHandle.free(acc, deviceArray[idx[0] * N + idx[1]]);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // free 1 allocation from 1 thread for N*N pointers
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                ScatterAllocator::AllocatorHandle allocHandle) {
                allocHandle.free(acc, deviceArray);
            },
            scatterAlloc.getAllocatorHandle()));
}

void test3D()
{
    using Dim = alpaka::dim::DimInt<3>;
    // using Acc = alpaka::acc::AccCpuThreads<Dim, Idx>;
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Idx>;

    using ScatterAllocator = mallocMC::Allocator<
        Acc,
        mallocMC::CreationPolicies::Scatter<ScatterConfig, ScatterHashParams>,
        mallocMC::DistributionPolicies::Noop,
        // mallocMC::DistributionPolicies::XMallocSIMD<DistributionConfig>,
        mallocMC::OOMPolicies::ReturnNull,
        // mallocMC::ReservePoolPolicies::SimpleMalloc,
        mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
        mallocMC::AlignmentPolicies::Shrink<AlignmentConfig>>;

    const auto dev
        = alpaka::pltf::getDevByIdx<alpaka::pltf::Pltf<alpaka::dev::Dev<Acc>>>(
            0);
    auto queue = alpaka::queue::Queue<Acc, alpaka::queue::Blocking>{dev};

    constexpr auto N = 16;

    ScatterAllocator scatterAlloc(dev, queue, 1024U * 1024U); // 1 MiB

    // make 1 allocation from 1 thread for N*N*N pointers
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                int N,
                ScatterAllocator::AllocatorHandle allocHandle) {
                deviceArray = (int **)allocHandle.malloc(
                    acc, sizeof(int *) * N * N * N);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // make N*N*N allocations from N*N*N threads for ints
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{N}, Idx{N}, Idx{N}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                int N,
                ScatterAllocator::AllocatorHandle allocHandle) {
                const auto idx
                    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                deviceArray[idx[0] * N * N + idx[1] * N + idx[0]]
                    = (int *)allocHandle.malloc(acc, sizeof(int));
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    const auto slots = scatterAlloc.getAvailableSlots(dev, queue, sizeof(int));
    const auto heapInfo = scatterAlloc.getHeapLocations().at(0);
    std::cout << "slots: " << slots << " heap size: " << heapInfo.size << '\n';

    // free N*N*N allocations from N*N*N threads for ints
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{N}, Idx{N}, Idx{N}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                int N,
                ScatterAllocator::AllocatorHandle allocHandle) {
                const auto idx
                    = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                allocHandle.free(
                    acc, deviceArray[idx[0] * N * N + idx[1] * N + idx[0]]);
            },
            N,
            scatterAlloc.getAllocatorHandle()));

    // free 1 allocation from 1 thread for N*N*N pointers
    alpaka::queue::enqueue(
        queue,
        alpaka::kernel::createTaskKernel<Acc>(
            alpaka::workdiv::WorkDivMembers<Dim, Idx>{
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}},
                alpaka::vec::Vec<Dim, Idx>{Idx{1}, Idx{1}, Idx{1}}},
            [] ALPAKA_FN_ACC(
                const Acc & acc,
                ScatterAllocator::AllocatorHandle allocHandle) {
                allocHandle.free(acc, deviceArray);
            },
            scatterAlloc.getAllocatorHandle()));
}

auto main(int argc, char ** argv) -> int
try
{
    test1D();
    test2D();
    test3D();

    return 0;
}
catch(const std::exception & e)
{
    std::cerr << e.what() << '\n';
}
