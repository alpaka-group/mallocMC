/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp
  https://www.hzdr.de/crp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at
              Carlchristian Eckert - c.eckert ( at ) hzdr.de

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

#include <alpaka/alpaka.hpp>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mallocMC
{
    template<int PSIZE>
    class __PointerEquivalent
    {
    public:
        using type = unsigned int;
    };
    template<>
    class __PointerEquivalent<8>
    {
    public:
        using type = unsigned long long;
    };

#if defined(__HIP__) || defined(__CUDA_ARCH__)
    constexpr auto warpSize = 32; // TODO
#else
    constexpr auto warpSize = 1;
#endif

    using PointerEquivalent
        = mallocMC::__PointerEquivalent<sizeof(char *)>::type;

    ALPAKA_FN_ACC inline auto laneid()
    {
#if defined(__HIP__) || defined(__HCC__)
        return __lane_id();
#elif defined(__CUDA_ARCH__)
        std::uint32_t mylaneid;
        asm("mov.u32 %0, %%laneid;" : "=r"(mylaneid));
        return mylaneid;
#else
        return 0u;
#endif
    }

    /** warp index within a multiprocessor
     *
     * Index of the warp within the multiprocessor at the moment of the query.
     * The result is volatile and can be different with each query.
     *
     * @return current index of the warp
     */
    ALPAKA_FN_ACC inline auto warpid()
    {
#if defined(__HIP__)
        // get wave id
        // https://github.com/ROCm-Developer-Tools/HIP/blob/f72a669487dd352e45321c4b3038f8fe2365c236/include/hip/hcc_detail/device_functions.h#L974-L1024
        return __builtin_amdgcn_s_getreg(GETREG_IMMED(3, 0, 4));
#elif defined(__HCC__)
        // workaround because wave id is not available for HCC
        return clock() % 8;
#elif defined(__CUDA_ARCH__)
        std::uint32_t mywarpid;
        asm("mov.u32 %0, %%warpid;" : "=r"(mywarpid));
        return mywarpid;
#else
        return 0u;
#endif
    }

    ALPAKA_FN_ACC inline auto smid()
    {
#if defined(__HIP__)
        return __smid();
#elif defined(__HCC__)
        // workaround because __smid is not available for HCC
        return clock() % 8;
#elif defined(__CUDA_ARCH__)
        std::uint32_t mysmid;
        asm("mov.u32 %0, %%smid;" : "=r"(mysmid));
        return mysmid;
#else
        return 0u;
#endif
    }

    ALPAKA_FN_ACC inline auto lanemask_lt()
    {
#if defined(__HIP__) || defined(__HCC__)
        return __lanemask_lt();
#elif defined(__CUDA_ARCH__)
        std::uint32_t lanemask;
        asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask));
        return lanemask;
#else
        return 0u;
#endif
    }

    ALPAKA_FN_ACC inline auto activemask()
    {
#if defined(__HIP__) || defined(__HCC__) || defined(__CUDA_ARCH__)
        return __activemask();
#else
        return 1u;
#endif
    }

    template<class T>
    ALPAKA_FN_HOST_ACC inline auto divup(T a, T b) -> T
    {
        return (a + b - 1) / b;
    }

    /** the maximal number threads per block
     *
     * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
     */
    struct MaxThreadsPerBlock
    {
        // valid for sm_2.X - sm_7.5
        // TODO alpaka
        static constexpr uint32_t value = 1024;
    };

    /** warp id within a cuda block
     *
     * The id is constant over the lifetime of the thread.
     * The id is not equal to warpid().
     *
     * @return warp id within the block
     */
    template<typename AlpakaAcc>
    ALPAKA_FN_ACC inline auto warpid_withinblock(const AlpakaAcc & acc)
        -> std::uint32_t
    {
        const auto localId = alpaka::idx::mapIdx<1>(
            alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc),
            alpaka::workdiv::getWorkDiv<alpaka::Block, alpaka::Threads>(
                acc))[0];
        return localId / warpSize;
    }

    ALPAKA_FN_ACC inline auto ffs(std::uint32_t mask) -> std::uint32_t
    {
#if defined(__HIP__) || defined(__HCC__) || defined(__CUDA_ARCH__)
        return ::__ffs(mask);
#else
        if(mask == 0)
            return 0;
        auto i = 1u;
        while((mask & 1) == 0)
        {
            mask >>= 1;
            i++;
        }
        return i;
#endif
    }

    ALPAKA_FN_ACC inline auto popc(std::uint32_t mask) -> std::uint32_t
    {
#if defined(__HIP__) || defined(__HCC__) || defined(__CUDA_ARCH__)
        return ::__popc(mask);
#else
        // cf.
        // https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
        std::uint32_t count = 0;
        while(mask)
        {
            count++;
            mask &= mask - 1;
        }
        return count;
#endif
    }
} // namespace mallocMC
