#pragma once

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
__device__ PolicyMallocType policyMallocGlobalObject;                          \
                                                                               \
__host__  void* initHeap(                                                      \
    size_t heapsize = 8U*1024U*1024U,                                          \
    PolicyMallocType &p = policyMallocGlobalObject                             \
    )                                                                          \
{                                                                              \
    return p.initHeap(heapsize);                                               \
}                                                                              \
__host__  void finalizeHeap(                                                   \
    PolicyMallocType &p = policyMallocGlobalObject                             \
    )                                                                          \
{                                                                              \
    p.finalizeHeap();                                                          \
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
__device__ void* pbMalloc(size_t t) __THROW                                    \
{                                                                              \
  return PolicyMalloc::malloc(t);                                              \
}                                                                              \
__device__ void  pbFree(void* p) __THROW                                       \
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
__device__ void* malloc(size_t t) __THROW                                      \
{                                                                              \
  return PolicyMalloc::policyMallocGlobalObject.alloc(t);                      \
}                                                                              \
__device__ void  free(void* p) __THROW                                         \
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
__device__ void* operator new(size_t t, PolicyMalloc::PolicyMallocType &h)     \
{                                                                              \
  return h.alloc(t);                                                           \
}                                                                              \
__device__ void* operator new[](size_t t, PolicyMalloc::PolicyMallocType &h)   \
{                                                                              \
  return h.alloc(t);                                                           \
}                                                                              \
__device__ void operator delete(void* p, PolicyMalloc::PolicyMallocType &h)    \
{                                                                              \
  h.dealloc(p);                                                                \
}                                                                              \
__device__ void operator delete[](void* p, PolicyMalloc::PolicyMallocType &h)  \
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
__device__ void* malloc(size_t t) __THROW                                      \
{                                                                              \
  return PolicyMalloc::malloc(t);                                              \
}                                                                              \
__device__ void  free(void* p) __THROW                                         \
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
POLICYMALLOC_PBMALLOC()

//POLICYMALLOC_OVERWRITE_NEW()

