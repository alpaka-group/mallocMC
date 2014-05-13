#pragma once

namespace PolicyMalloc{
namespace OOMPolicies{

  /**
   * @brief Throws a std::bad_alloc exception on OutOfMemory
   *
   * This OOMPolicy will throw a std::bad_alloc exception, if the accelerator
   * supports it. Currently, Nvidia CUDA does not support any form of exception
   * handling, therefore handleOOM() does not have any effect on these
   * accelerators. Using this policy on other types of accelerators that do not
   * support exceptions results in undefined behaviour.
   */
  class BadAllocException;
    
} //namespace OOMPolicies
} //namespace PolicyMalloc
