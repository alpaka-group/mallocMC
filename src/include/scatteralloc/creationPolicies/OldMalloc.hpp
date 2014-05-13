#pragma once


namespace PolicyMalloc{
namespace CreationPolicies{
    
  /**
   * @brief classic malloc/free behaviour known from CUDA
   *
   * This CreationPolicy implements the classic device-side malloc and free
   * system calls that is offered by CUDA-capable accelerator of compute
   * capability 2.0 and higher
   */
  class OldMalloc;

} //namespace CreationPolicies
} //namespace PolicyMalloc
