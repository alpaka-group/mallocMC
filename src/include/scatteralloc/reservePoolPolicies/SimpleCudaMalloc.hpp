#pragma once

namespace PolicyMalloc{
namespace ReservePoolPolicies{

  /**
   * @brief creates/allocates a fixed memory pool on the accelerator
   *
   * This ReservePoolPolicy will create a memory pool of a fixed size on the
   * accelerator by using a host-side call to cudaMalloc(). The pool is later
   * freed through cudaFree(). This can only be used with accelerators that
   * support CUDA and compute capability 2.0 or higher.
   */
  struct SimpleCudaMalloc;

} //namespace ReservePoolPolicies
} //namespace PolicyMalloc
