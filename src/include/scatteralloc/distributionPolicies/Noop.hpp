#pragma once


namespace PolicyMalloc{
namespace DistributionPolicies{

  /**
   * @brief a policy that does nothing
   *
   * This DistributionPolicy will not perform any distribution, but only return
   * its input (identity function)
   */
  class Noop;

} //namespace DistributionPolicies
} //namespace PolicyMalloc
