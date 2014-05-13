#pragma once

namespace PolicyMalloc{
namespace AlignmentPolicies{

  /**
   * @brief a policy that does nothing
   *
   * This AlignmentPolicy will not perform any distribution, but only return
   * its input (identity function)
   */
  class Noop;

} //namespace AlignmentPolicies
} //namespace PolicyMalloc
