/*
  ScatterAlloc: Massively Parallel Dynamic Memory Allocation for the GPU.
  http://www.icg.tugraz.at/project/mvp
  https://www.hzdr.de/crp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology
  Copyright (C) 2014 Institute of Radiation Physics,
                     Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
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

#include "policy_malloc_prefixes.hpp"

/** Creates a global object of a policyBased memory allocator
 *
 * Will create a global object of the supplied type and use it to generate
 * global functions that use this type internally. This should be done after
 * defining a new policy based memory allocator with a typedef.
 */
#define POLICYMALLOC_GLOBAL_FUNCTIONS(POLICYMALLOC_USER_DEFINED_TYPENAME)      \
namespace PolicyMalloc{                                                        \
  typedef POLICYMALLOC_USER_DEFINED_TYPENAME PolicyMallocType;                 \
                                                                               \
PMMA_ACCELERATOR PolicyMallocType policyMallocGlobalObject;                    \
                                                                               \
PMMA_HOST void* initHeap(                                                      \
    size_t heapsize = 8U*1024U*1024U,                                          \
    PolicyMallocType &p = policyMallocGlobalObject                             \
    )                                                                          \
{                                                                              \
    return p.initHeap(heapsize);                                               \
}                                                                              \
PMMA_HOST void finalizeHeap(                                                   \
    PolicyMallocType &p = policyMallocGlobalObject                             \
    )                                                                          \
{                                                                              \
    p.finalizeHeap();                                                          \
}                                                                              \
} /* end namespace PolicyMalloc */


/** Provides the easily accessible functions getAvaliableSlots
 *
 * Will use the global object defined by POLICYMALLOC_SET_ALLOCATOR_TYPE and
 * use it to generate global functions that use this type internally.
 */
#define POLICYMALLOC_AVAILABLESLOTS()                                          \
namespace PolicyMalloc{                                                        \
PMMA_HOST PMMA_ACCELERATOR                                                     \
unsigned getAvailableSlots(                                                    \
    size_t slotSize,                                                           \
    PolicyMallocType &p = policyMallocGlobalObject){                           \
    return p.getAvailableSlots(slotSize);                                      \
}                                                                              \
PMMA_HOST PMMA_ACCELERATOR                                                     \
bool providesAvailableSlots(){                                                 \
    return Traits<PolicyMallocType>::providesAvailableSlots;                   \
}                                                                              \
} /* end namespace PolicyMalloc */



/** Create the functions pbMalloc() and pbFree() inside a namespace
 *
 * This allows to use a function without bothering with name-clashes when
 * including a namespace in the global scope. It will call the namespaced
 * version of malloc() inside.
 */
#define POLICYMALLOC_PBMALLOC()                                                \
namespace PolicyMalloc{                                                        \
PMMA_ACCELERATOR                                                               \
void* pbMalloc(size_t t) __THROW                                               \
{                                                                              \
  return PolicyMalloc::malloc(t);                                              \
}                                                                              \
PMMA_ACCELERATOR                                                               \
void  pbFree(void* p) __THROW                                                  \
{                                                                              \
  PolicyMalloc::free(p);                                                       \
}                                                                              \
} /* end namespace PolicyMalloc */


/** Create the functions malloc() and free() inside a namespace
 *
 * This allows for a peaceful coexistence between different functions called
 * "malloc" or "free". This is useful when using a policy that contains a call
 * to the original device-side malloc() from CUDA.
 */
#define POLICYMALLOC_MALLOC()                                                  \
namespace PolicyMalloc{                                                        \
PMMA_ACCELERATOR                                                               \
void* malloc(size_t t) __THROW                                                 \
{                                                                              \
  return PolicyMalloc::policyMallocGlobalObject.alloc(t);                      \
}                                                                              \
PMMA_ACCELERATOR                                                               \
void  free(void* p) __THROW                                                    \
{                                                                              \
  PolicyMalloc::policyMallocGlobalObject.dealloc(p);                           \
}                                                                              \
} /* end namespace PolicyMalloc */



/** Override/replace the global implementation of placement new/delete on CUDA
 *
 * These overrides are for device-side new and delete and need a pointer to the
 * memory-allocator object on device (this will be mostly useful when using
 * more advanced techniques and managing your own global object instead of
 * using the provided macros).
 *
 * @param h the allocator as returned by initHeap()
 */
#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 200
#define POLICYMALLOC_OVERWRITE_NEW()                                           \
PMMA_ACCELERATOR                                                               \
void* operator new(size_t t, PolicyMalloc::PolicyMallocType &h)                \
{                                                                              \
  return h.alloc(t);                                                           \
}                                                                              \
PMMA_ACCELERATOR                                                               \
void* operator new[](size_t t, PolicyMalloc::PolicyMallocType &h)              \
{                                                                              \
  return h.alloc(t);                                                           \
}                                                                              \
PMMA_ACCELERATOR                                                               \
void operator delete(void* p, PolicyMalloc::PolicyMallocType &h)               \
{                                                                              \
  h.dealloc(p);                                                                \
}                                                                              \
PMMA_ACCELERATOR                                                               \
void operator delete[](void* p, PolicyMalloc::PolicyMallocType &h)             \
{                                                                              \
  h.dealloc(p);                                                                \
}
#endif
#endif



/** Override/replace the global implementation of malloc/free on CUDA devices
 *
 * Attention: This will also replace "new", "new[]", "delete" and "delete[]",
 * since CUDA uses the same malloc/free functions for that. Needs at least
 * ComputeCapability 2.0
 */
#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 200
#define POLICYMALLOC_OVERWRITE_MALLOC()                                        \
PMMA_ACCELERATOR                                                               \
void* malloc(size_t t) __THROW                                                 \
{                                                                              \
  return PolicyMalloc::malloc(t);                                              \
}                                                                              \
PMMA_ACCELERATOR                                                               \
void  free(void* p) __THROW                                                    \
{                                                                              \
  PolicyMalloc::free(p);                                                       \
}
#endif
#endif



/* if the defines do not exist (wrong CUDA version etc),
 * create at least empty defines
 */
#ifndef POLICYMALLOC_PBMALLOC
#define POLICYMALLOC_PBMALLOC()
#endif

#ifndef POLICYMALLOC_MALLOC
#define POLICYMALLOC_MALLOC()
#endif

#ifndef POLICYMALLOC_OVERWRITE_NEW
#define POLICYMALLOC_OVERWRITE_NEW()
#endif

#ifndef POLICYMALLOC_OVERWRITE_MALLOC
#define POLICYMALLOC_OVERWRITE_MALLOC()
#endif



/** Set up the global variables and functions
 *
 * This Macro should be called with the type of a newly defined allocator. It
 * will create a global object of that allocator and the necessary functions to
 * initialize and allocate memory.
 */
#define POLICYMALLOC_SET_ALLOCATOR_TYPE(POLICYMALLOC_USER_DEFINED_TYPE)        \
POLICYMALLOC_GLOBAL_FUNCTIONS(POLICYMALLOC_USER_DEFINED_TYPE)                  \
POLICYMALLOC_MALLOC()                                                          \
POLICYMALLOC_PBMALLOC()                                                        \
POLICYMALLOC_AVAILABLESLOTS()

//POLICYMALLOC_OVERWRITE_NEW()

