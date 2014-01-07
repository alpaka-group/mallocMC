/*
  ScatterAlloc: Massively Parallel Dynamic Memory Allocation for the GPU.
  http://www.icg.tugraz.at/project/mvp

  Copyright (C) 2012 Institute for Computer Graphics and Vision,
                     Graz University of Technology

  Author(s):  Markus Steinberger - steinberger ( at ) icg.tugraz.at
              Michael Kenzel - kenzel ( at ) icg.tugraz.at

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


#ifndef INCLUDED_GPUTOOLS_UTILS_H
#define INCLUDED_GPUTOOLS_UTILS_H

#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <string>
#include <sstream>
#include <stdexcept>


namespace CUDA
{
  class error : public std::runtime_error
  {
  private:
    static std::string genErrorString(cudaError error, const char* file, int line)
    {
      std::ostringstream msg;
      msg << file << '(' << line << "): error: " << cudaGetErrorString(error);
      return msg.str();
    }
  public:
    error(cudaError error, const char* file, int line)
      : runtime_error(genErrorString(error, file, line))
    {
    }

    error(cudaError error)
      : runtime_error(cudaGetErrorString(error))
    {
    }

    error(const std::string& msg)
      : runtime_error(msg)
    {
    }
  };

  inline void checkError(cudaError error, const char* file, int line)
  {
#ifdef _DEBUG
    if (error != cudaSuccess)
      throw CUDA::error(error, file, line);
#endif
  }

  inline void checkError(const char* file, int line)
  {
    checkError(cudaGetLastError(), file, line);
  }

  inline void checkError()
  {
#ifdef _DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
      throw CUDA::error(error);
#endif
  }

#define CUDA_CHECKED_CALL(call) CUDA::checkError(call, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CUDA::checkError(__FILE__, __LINE__)
}


#define warp_serial                                    \
  for (uint32 __mask = __ballot(1),                    \
            __num = __popc(__mask),                    \
            __lanemask = GPUTools::lanemask_lt(),      \
            __local_id = __popc(__lanemask & __mask),  \
            __active = 0;                              \
       __active < __num;                               \
       ++__active)                                     \
    if (__active == __local_id)


namespace GPUTools
{    
  typedef unsigned int uint32;

  template<int PSIZE>
  class __PointerEquivalent
  {
  public:
    typedef unsigned int type;
  };
  template<>
  class __PointerEquivalent<8>
  {
  public:
    typedef unsigned long long int type;
  };

  typedef GPUTools::__PointerEquivalent<sizeof(char*)>::type PointerEquivalent;


  __device__ inline uint32 laneid()
  {
    uint32 mylaneid;
    asm("mov.u32 %0, %laneid;" : "=r" (mylaneid));
    return mylaneid;
  }

  __device__ inline uint32 warpid()
  {
    uint32 mywarpid;
    asm("mov.u32 %0, %warpid;" : "=r" (mywarpid));
    return mywarpid;
  }
  __device__ inline uint32 nwarpid()
  {
    uint32 mynwarpid;
    asm("mov.u32 %0, %nwarpid;" : "=r" (mynwarpid));
    return mynwarpid;
  }

  __device__ inline uint32 smid()
  {
    uint32 mysmid;
    asm("mov.u32 %0, %smid;" : "=r" (mysmid));
    return mysmid;
  }

  __device__ inline uint32 nsmid()
  {
    uint32 mynsmid;
    asm("mov.u32 %0, %nsmid;" : "=r" (mynsmid));
    return mynsmid;
  }
  __device__ inline uint32 lanemask()
  {
    uint32 lanemask;
    asm("mov.u32 %0, %lanemask_eq;" : "=r" (lanemask));
    return lanemask;
  }

  __device__ inline uint32 lanemask_le()
  {
    uint32 lanemask;
    asm("mov.u32 %0, %lanemask_le;" : "=r" (lanemask));
    return lanemask;
  }

  __device__ inline uint32 lanemask_lt()
  {
    uint32 lanemask;
    asm("mov.u32 %0, %lanemask_lt;" : "=r" (lanemask));
    return lanemask;
  }

  __device__ inline uint32 lanemask_ge()
  {
    uint32 lanemask;
    asm("mov.u32 %0, %lanemask_ge;" : "=r" (lanemask));
    return lanemask;
  }

  __device__ inline uint32 lanemask_gt()
  {
    uint32 lanemask;
    asm("mov.u32 %0, %lanemask_gt;" : "=r" (lanemask));
    return lanemask;
  }

  template<class T>
  __host__ __device__ inline T divup(T a, T b) { return (a + b - 1)/b; } 

}



#endif  // INCLUDED_GPUTOOLS_UTILS_H
