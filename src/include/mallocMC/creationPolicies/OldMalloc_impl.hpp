/*
  mallocMC: Memory Allocator for Many Core Architectures.

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

#include "OldMalloc.hpp"

#include <alpaka/core/Common.hpp>
#include <cstdint>

namespace mallocMC
{
    namespace CreationPolicies
    {
        class OldMalloc
        {
            using uint32 = std::uint32_t;

        public:
            static constexpr auto providesAvailableSlots = false;

            ALPAKA_FN_ACC auto create(uint32 bytes) const -> void *
            {
                return ::malloc(static_cast<size_t>(bytes));
            }

            template <typename AlpakaAcc>
            ALPAKA_FN_ACC void destroy(const AlpakaAcc& /*acc*/, void * mem) const
            {
                ::free(mem);
            }

            ALPAKA_FN_ACC auto isOOM(void * p, size_t s) const -> bool
            {
                return s != 0 && (p == nullptr);
            }

            template<typename AlpakaQueue, typename T>
            static auto
            initHeap(AlpakaQueue & queue, T * dAlloc, void *, size_t) -> void *
            {
                return dAlloc;
            }

            static auto classname() -> std::string
            {
                return "OldMalloc";
            }
        };

    } // namespace CreationPolicies
} // namespace mallocMC
