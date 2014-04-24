#pragma once

#include <boost/mpl/bool.hpp>
#include <boost/mpl/int.hpp>

namespace PolicyMalloc{
namespace CreationPolicies{
namespace ScatterConf{
  struct DefaultScatterConfig{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
  };

  struct DefaultScatterHashingParams{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
  };  
}

  template<
  class T_Config = ScatterConf::DefaultScatterConfig,
  class T_Hashing = ScatterConf::DefaultScatterHashingParams
  >
  class Scatter;

}// namespace CreationPolicies
}// namespace PolicyMalloc
