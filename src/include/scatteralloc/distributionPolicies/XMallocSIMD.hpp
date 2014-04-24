#pragma once

#include <boost/mpl/int.hpp>

namespace PolicyMalloc{
namespace DistributionPolicies{
    
  namespace XMallocSIMDConf{
    struct DefaultXMallocConfig{
      typedef boost::mpl::int_<4096>     pagesize;
    };  
  }

  template<class T_Config=XMallocSIMDConf::DefaultXMallocConfig>
  class XMallocSIMD;


} //namespace DistributionPolicies
} //namespace PolicyMalloc
