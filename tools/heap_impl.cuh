/*
  ScatterAlloc: Massively Parallel Dynamic Memory Allocation for the GPU.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at

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

#ifndef HEAP_IMPL_CUH
#define HEAP_IMPL_CUH

#include "tools/heap.cuh"
#ifndef HEAPARGS
typedef GPUTools::DeviceHeap<> heap_t;
#else
typedef  GPUTools::DeviceHeap<HEAPARGS> heap_t;
#endif

__device__  heap_t theHeap;

void* initHeap(size_t memsize = 8*1024U*1024U)
{
  void* pool;
  heap_t* heap;
  CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&heap,theHeap));
  CUDA_CHECKED_CALL(cudaMalloc(&pool, memsize));
  GPUTools::initHeap<<<1,256>>>(heap, pool, memsize);
  return pool;
}




#ifdef __CUDACC__
#ifdef OVERWRITE_MALLOC
#if __CUDA_ARCH__ >= 200
__device__ void* malloc(size_t t)
{
  return theHeap.alloc(t);
}
__device__ void  free(void* p)
{
  theHeap.dealloc(p);
}
#define sNew new
#define sNewA new
#define sDelete(what) delete what
#define sDeleteA(what) delete what
#endif
#else
#define sNew new(theHeap)
#define sNewA new(theHeap)
#define sDelete(what) theHeap.deleteS(what)
#define sDeleteA(what) theHeap.deleteA(what)
template<uint pagesize, uint accessblocks, uint regionsize, uint wastefactor, bool use_coalescing, bool resetfreedpages>
__device__ void* operator new(size_t bytes, GPUTools::DeviceHeap<pagesize, accessblocks,  regionsize, wastefactor, use_coalescing, resetfreedpages> &h )
{
  return h.alloc(bytes);
}
template<uint pagesize, uint accessblocks, uint regionsize, uint wastefactor, bool use_coalescing, bool resetfreedpages>
__device__ void* operator new[](size_t bytes, GPUTools::DeviceHeap<pagesize, accessblocks,  regionsize, wastefactor, use_coalescing, resetfreedpages> &h )
{
  return h.alloc(bytes);
}
template<uint pagesize, uint accessblocks, uint regionsize, uint wastefactor, bool use_coalescing, bool resetfreedpages>
__device__ void operator delete(void* mem, GPUTools::DeviceHeap<pagesize, accessblocks,  regionsize, wastefactor, use_coalescing, resetfreedpages> &h )
{
  h.dealloc(mem);
}
template<uint pagesize, uint accessblocks, uint regionsize, uint wastefactor,  bool use_coalescing, bool resetfreedpages>
__device__ void operator delete[](void* mem, GPUTools::DeviceHeap<pagesize, accessblocks,  regionsize, wastefactor, use_coalescing, resetfreedpages> &h )
{
  h.dealloc(mem);
}
#endif 

#endif //__CUDACC__

#endif //HEAP_IMPL_CUH
