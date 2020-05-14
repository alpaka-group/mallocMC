/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 - 2015 Institute of Radiation Physics,
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

#include "device_allocator.hpp"
#include "mallocMC_allocator_handle.hpp"
#include "mallocMC_constraints.hpp"
#include "mallocMC_prefixes.hpp"
#include "mallocMC_traits.hpp"
#include "mallocMC_utils.hpp"

#include <cstdint>
#include <sstream>
#include <tuple>
#include <vector>

namespace mallocMC
{
    namespace detail
    {
        template<typename T_Allocator, bool T_providesAvailableSlots>
        struct GetAvailableSlotsIfAvailHost
        {
            MAMC_HOST static auto getAvailableSlots(size_t, T_Allocator &)
                -> unsigned
            {
                return 0;
            }
        };

        template<class T_Allocator>
        struct GetAvailableSlotsIfAvailHost<T_Allocator, true>
        {
            MAMC_HOST
            static auto getAvailableSlots(size_t slotSize, T_Allocator & alloc)
                -> unsigned
            {
                return T_Allocator::CreationPolicy::getAvailableSlotsHost(
                    slotSize, alloc.getAllocatorHandle().devAllocator);
            }
        };
    } // namespace detail

    struct HeapInfo
    {
        void * p;
        size_t size;
    };

    /**
     * @brief "HostClass" that combines all policies to a useful allocator
     *
     * This class implements the necessary glue-logic to form an actual
     * allocator from the provided policies. It implements the public interface
     * and executes some constraint checking based on an instance of the class
     * PolicyConstraints.
     *
     * @tparam T_CreationPolicy The desired type of a CreationPolicy
     * @tparam T_DistributionPolicy The desired type of a DistributionPolicy
     * @tparam T_OOMPolicy The desired type of a OOMPolicy
     * @tparam T_ReservePoolPolicy The desired type of a ReservePoolPolicy
     * @tparam T_AlignmentPolicy The desired type of a AlignmentPolicy
     */
    template<
        typename T_CreationPolicy,
        typename T_DistributionPolicy,
        typename T_OOMPolicy,
        typename T_ReservePoolPolicy,
        typename T_AlignmentPolicy>
    class Allocator :
            public PolicyConstraints<
                T_CreationPolicy,
                T_DistributionPolicy,
                T_OOMPolicy,
                T_ReservePoolPolicy,
                T_AlignmentPolicy>
    {
        using uint32 = std::uint32_t;

    public:
        using CreationPolicy = T_CreationPolicy;
        using DistributionPolicy = T_DistributionPolicy;
        using OOMPolicy = T_OOMPolicy;
        using ReservePoolPolicy = T_ReservePoolPolicy;
        using AlignmentPolicy = T_AlignmentPolicy;
        using HeapInfoVector = std::vector<HeapInfo>;
        using DevAllocator = DeviceAllocator<
            CreationPolicy,
            DistributionPolicy,
            OOMPolicy,
            AlignmentPolicy>;
        using AllocatorHandle = AllocatorHandleImpl<Allocator>;

    private:
        AllocatorHandle allocatorHandle;
        HeapInfo heapInfos;

        /** allocate heap memory
         *
         * @param size number of bytes
         */
        MAMC_HOST
        void alloc(size_t size)
        {
            void * pool = ReservePoolPolicy::setMemPool(size);
            std::tie(pool, size) = AlignmentPolicy::alignPool(pool, size);
            DevAllocator * devAllocatorPtr;
            cudaMalloc((void **)&devAllocatorPtr, sizeof(DevAllocator));
            CreationPolicy::initHeap(devAllocatorPtr, pool, size);

            allocatorHandle.devAllocator = devAllocatorPtr;
            heapInfos.p = pool;
            heapInfos.size = size;
        }

        /** free all data structures
         *
         * Free all allocated memory.
         * After this call the instance is an in invalid state
         */
        MAMC_HOST
        void free()
        {
            cudaFree(allocatorHandle.devAllocator);
            ReservePoolPolicy::resetMemPool(heapInfos.p);
            allocatorHandle.devAllocator = nullptr;
            heapInfos.size = 0;
            heapInfos.p = nullptr;
        }

        /* forbid to copy the allocator */
        MAMC_HOST
        Allocator(const Allocator &);

    public:
        MAMC_HOST
        Allocator(size_t size = 8U * 1024U * 1024U) : allocatorHandle(nullptr)
        {
            alloc(size);
        }

        MAMC_HOST
        ~Allocator()
        {
            free();
        }

        /** destroy current heap data and resize the heap
         *
         * @param size number of bytes
         */
        MAMC_HOST
        void destructiveResize(size_t size)
        {
            free();
            alloc(size);
        }

        MAMC_HOST
        auto getAllocatorHandle() -> AllocatorHandle
        {
            return allocatorHandle;
        }

        MAMC_HOST
        operator AllocatorHandle()
        {
            return getAllocatorHandle();
        }

        MAMC_HOST static auto info(std::string linebreak = " ") -> std::string
        {
            std::stringstream ss;
            ss << "CreationPolicy:      " << CreationPolicy::classname()
               << "    " << linebreak;
            ss << "DistributionPolicy:  " << DistributionPolicy::classname()
               << "" << linebreak;
            ss << "OOMPolicy:           " << OOMPolicy::classname()
               << "         " << linebreak;
            ss << "ReservePoolPolicy:   " << ReservePoolPolicy::classname()
               << " " << linebreak;
            ss << "AlignmentPolicy:     " << AlignmentPolicy::classname()
               << "   " << linebreak;
            return ss.str();
        }

        // polymorphism over the availability of getAvailableSlots for calling
        // from the host
        MAMC_HOST
        auto getAvailableSlots(size_t slotSize) -> unsigned
        {
            slotSize = AlignmentPolicy::applyPadding(slotSize);
            return detail::GetAvailableSlotsIfAvailHost<
                Allocator,
                Traits<Allocator>::providesAvailableSlots>::
                getAvailableSlots(slotSize, *this);
        }

        MAMC_HOST
        auto getHeapLocations() -> HeapInfoVector
        {
            HeapInfoVector v;
            v.push_back(heapInfos);
            return v;
        }
    };

} // namespace mallocMC
