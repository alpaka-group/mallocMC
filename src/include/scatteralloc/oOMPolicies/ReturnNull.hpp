#pragma once

namespace PolicyMalloc{
namespace OOMPolicies{

  /**
   * @brief Returns a NULL pointer on OutOfMemory conditions
   *
   * This OOMPolicy will return NULL, if handleOOM() is called.
   */
  class ReturnNull;
    
} //namespace OOMPolicies
} //namespace PolicyMalloc
