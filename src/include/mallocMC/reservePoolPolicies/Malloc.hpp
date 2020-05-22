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

#pragma once

#include "SimpleCudaMalloc.hpp"
#include "SimpleMalloc.hpp"

#include <alpaka/alpaka.hpp>
#include <cstdlib>
#include <string>

namespace mallocMC
{
    namespace ReservePoolPolicies
    {
        template<typename AlpakaAcc>
        struct Malloc : SimpleMalloc
        {};
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
        template<typename Dim, typename Idx>
        struct Malloc<alpaka::acc::AccGpuCudaRt<Dim, Idx>> : SimpleCudaMalloc
        {};
#endif
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
        template<typename Dim, typename Idx>
        struct Malloc<alpaka::acc::AccGpuHipRt<Dim, Idx>> : SimpleCudaMalloc
        {};
#endif
    } // namespace ReservePoolPolicies
} // namespace mallocMC