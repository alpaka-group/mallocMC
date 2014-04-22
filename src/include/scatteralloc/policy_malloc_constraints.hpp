#pragma once

#include "creationPolicies/Scatter.hpp"
#include "distributionPolicies/XMallocSIMD.hpp"
#include <boost/mpl/assert.hpp>

namespace PolicyMalloc{

  /** The default PolicyChecker (does always succeed)
   */
  template<typename Policy1, typename Policy2>
  struct PolicyChecker{};


  /** Enforces constraints on policies or combinations of polices
   * 
   * Uses template specialization of PolicyChecker
   */
  template < 
     typename T_CreationPolicy, 
     typename T_DistributionPolicy, 
     typename T_OOMPolicy, 
     typename T_GetHeapPolicy,
     typename T_AlignmentPolicy
       >
  class PolicyConstraints{
      PolicyChecker<T_CreationPolicy, T_DistributionPolicy> c;
  };


  /** Scatter and XMallocSIMD need the same pagesize!
   *
   * This constraint ensures that if the CreationPolicy "Scatter" and the
   * DistributionPolicy "XMallocSIMD" are selected, they are configured to use
   * the same value for their "pagesize"-parameter.
   */
  template<typename x, typename y, typename z >
  struct PolicyChecker<
    typename CreationPolicies::Scatter<x,y>,
    typename DistributionPolicies::XMallocSIMD<z> 
  >{
    BOOST_MPL_ASSERT_MSG(x::pagesize::value == z::pagesize::value,
        Pagesize_must_be_the_same_when_combining_Scatter_and_XMallocSIMD, () );
  };

}//namespace PolicyMalloc
