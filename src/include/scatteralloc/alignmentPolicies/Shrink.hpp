#pragma once

#include <boost/mpl/int.hpp>

namespace PolicyMalloc{
namespace AlignmentPolicies{

namespace ShrinkConfig{
  struct DefaultShrinkConfig{
    typedef boost::mpl::int_<16> dataAlignment;
  };
}

  template<typename T_Config = ShrinkConfig::DefaultShrinkConfig>
  class Shrink;

} //namespace AlignmentPolicies
} //namespace PolicyMalloc
