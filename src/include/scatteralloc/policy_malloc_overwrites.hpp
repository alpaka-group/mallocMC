#pragma once
#define POLICY_MALLOC_GLOBAL_FUNCTIONS_INTERNAL(POLICY_MALLOC_USER_DEFINED_TYPENAME_INTERNAL)      \
namespace PolicyMalloc{                                                        \
  typedef POLICY_MALLOC_USER_DEFINED_TYPENAME_INTERNAL PolicyMallocType;       \
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
} //end namespace PolicyMalloc


#define POLICY_MALLOC_MEMORY_ALLOCATOR_FUNCTIONS()                             \
namespace PolicyMalloc{                                                        \
__device__ void* pbMalloc(size_t t) __THROW                                    \
{                                                                              \
  return PolicyMalloc::policyMallocGlobalObject.alloc(t);                      \
}                                                                              \
__device__ void  pbFree(void* p) __THROW                                       \
{                                                                              \
  PolicyMalloc::policyMallocGlobalObject.dealloc(p);                           \
}                                                                              \
} //end namespace PolicyMalloc


#define POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_NAMESPACE()                      \
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


#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 200
#define POLICY_MALLOC_MEMORY_ALLOCATOR_NEW_OVERWRITE()                         \
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


#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 200
#define POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE()                      \
__device__ void* malloc(size_t t) __THROW                                      \
{                                                                              \
  return PolicyMalloc::policyMallocGlobalObject.alloc(t);                      \
}                                                                              \
__device__ void  free(void* p) __THROW                                         \
{                                                                              \
  PolicyMalloc::policyMallocGlobalObject.dealloc(p);                           \
}
#endif
#endif


#ifndef POLICY_MALLOC_MEMORY_ALLOCATOR_FUNCTIONS
#define POLICY_MALLOC_MEMORY_ALLOCATOR_FUNCTIONS()
#endif

#ifndef POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_NAMESPACE
#define POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_NAMESPACE()
#endif

#ifndef POLICY_MALLOC_MEMORY_ALLOCATOR_NEW_OVERWRITE
#define POLICY_MALLOC_MEMORY_ALLOCATOR_NEW_OVERWRITE()
#endif

#ifndef POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE
#define POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE()
#endif

#define SET_ACCELERATOR_MEMORY_ALLOCATOR_TYPE(POLICY_MALLOC_USER_DEFINED_TYPE)\
POLICY_MALLOC_GLOBAL_FUNCTIONS_INTERNAL(POLICY_MALLOC_USER_DEFINED_TYPE)\
POLICY_MALLOC_MEMORY_ALLOCATOR_FUNCTIONS()\
POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_NAMESPACE()

//POLICY_MALLOC_MEMORY_ALLOCATOR_NEW_OVERWRITE()

