/*
  mallocMC: Memory Allocator for Many Core Architectures.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Rene Widera - r.widera ( at ) hzdr.de
              Axel Huebl - a.huebl ( at ) hzdr.de
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

#include "../mallocMC_utils.hpp"
#include "XMallocSIMD.hpp"

#include <alpaka/alpaka.hpp>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

namespace mallocMC
{
    namespace DistributionPolicies
    {
        template<typename T_Config>
        class XMallocSIMD
        {
        private:
            using uint32 = std::uint32_t;
            bool can_use_coalescing;
            uint32 warpid;
            uint32 myoffset;
            uint32 threadcount;
            uint32 req_size;

        public:
            using Properties = T_Config;

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC XMallocSIMD(const AlpakaAcc & acc) :
                    can_use_coalescing(false),
                    warpid(warpid_withinblock(acc)),
                    myoffset(0),
                    threadcount(0),
                    req_size(0)
            {}

        private:
/** Allow for a hierarchical validation of parameters:
 *
 * shipped default-parameters (in the inherited struct) have lowest precedence.
 * They will be overridden by a given configuration struct. However, even the
 * given configuration struct can be overridden by compile-time command line
 * parameters (e.g. -D MALLOCMC_DP_XMALLOCSIMD_PAGESIZE 1024)
 *
 * default-struct < template-struct < command-line parameter
 */
#ifndef MALLOCMC_DP_XMALLOCSIMD_PAGESIZE
#define MALLOCMC_DP_XMALLOCSIMD_PAGESIZE (Properties::pagesize)
#endif
            static constexpr uint32 pagesize = MALLOCMC_DP_XMALLOCSIMD_PAGESIZE;

        public:
            static constexpr uint32 _pagesize = pagesize;

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto collect(const AlpakaAcc & acc, uint32 bytes)
                -> uint32
            {
                can_use_coalescing = false;
                myoffset = 0;
                threadcount = 0;

                // init with initial counter
                auto & warp_sizecounter = alpaka::block::shared::st::allocVar<
                    std::uint32_t[MaxThreadsPerBlock::value / warpSize],
                    __COUNTER__>(acc);
                warp_sizecounter[warpid] = 16;

                // second half: make sure that all coalesced allocations can fit
                // within one page necessary for offset calculation
                const bool coalescible = bytes > 0 && bytes < (pagesize / 32);
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
                threadcount = popc(__ballot_sync(__activemask(), coalescible));
#else
                threadcount = 1; // TODO
#endif
                if(coalescible && threadcount > 1)
                {
                    myoffset
                        = alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(
                            acc, &warp_sizecounter[warpid], bytes);
                    can_use_coalescing = true;
                }

                req_size = bytes;
                if(can_use_coalescing)
                    req_size = (myoffset == 16) ? warp_sizecounter[warpid] : 0;

                return req_size;
            }

            template<typename AlpakaAcc>
            ALPAKA_FN_ACC auto
            distribute(const AlpakaAcc & acc, void * allocatedMem) -> void *
            {
                auto & warp_res = alpaka::block::shared::st::allocVar<
                    char * [MaxThreadsPerBlock::value / warpSize],
                    __COUNTER__>(acc);

                char * myalloc = (char *)allocatedMem;
                if(req_size && can_use_coalescing)
                {
                    warp_res[warpid] = myalloc;
                    if(myalloc != 0)
                        *(uint32 *)myalloc = threadcount;
                }
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
                __threadfence_block();
#else
                std::atomic_thread_fence(
                    std::memory_order::memory_order_seq_cst);
#endif
                void * myres = myalloc;
                if(can_use_coalescing)
                {
                    if(warp_res[warpid] != 0)
                        myres = warp_res[warpid] + myoffset;
                    else
                        myres = 0;
                }
                return myres;
            }

            ALPAKA_FN_HOST
            static auto classname() -> std::string
            {
                std::stringstream ss;
                ss << "XMallocSIMD[" << pagesize << "]";
                return ss.str();
            }
        };

    } // namespace DistributionPolicies

} // namespace mallocMC
