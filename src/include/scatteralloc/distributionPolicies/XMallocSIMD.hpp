#pragma once

namespace PolicyMalloc{
namespace DistributionPolicies{
    
  template<class T_Dummy>
  class XMallocSIMD2;

  typedef XMallocSIMD2<void> XMallocSIMD;

} //namespace DistributionPolicies
} //namespace PolicyMalloc
