#pragma once

namespace PolicyMalloc{
namespace ReservePoolPolicies{

  /**
   * @brief set CUDA internal heap for device-side malloc calls
   *
   * This ReservePoolPolicy is intended for use with CUDA capable accelerators
   * that support at least compute capability 2.0. It should be used in
   * conjunction with a CreationPolicy that actually requires the CUDA-internal
   * heap to be sized by calls to cudaDeviceSetLimit()
   */
  struct CudaSetLimits;

} //namespace ReservePoolPolicies
} //namespace PolicyMalloc
