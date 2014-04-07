#pragma once

namespace PolicyMalloc{
namespace AlignmentPolicies{

  template<typename T_Dummy>
  class Shrink2;

  typedef Shrink2<void> Shrink;

} //namespace AlignmentPolicies
} //namespace PolicyMalloc
