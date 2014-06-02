Change Log / Release Log for mallocMC
================================================================

2.0.0crp
-------------
**Date:** 2014-06-02

This release introduces mallocMC, which contains the previous algorithm and
much code from ScatterAlloc 1.0.2crp. The project was renamed due to massive
restructurization and because the code uses ScatterAlloc as a reference
algorithm, but can be extended to include other allocators in the future.
We closed all issues documented in
[Milestone *Get Lib ready for PIConGPU*](https://github.com/ComputationalRadiationPhysics/mallocMC/issues?milestone=2&state=closed)

### Changes to ScatterAlloc 1.0.2crp

**Features**
 - completely split into policies #17
 - configuration through structs instead of macro #17
 - function `getAvailableSlots()` #5
 - selectable data alignment #14
 - function `finalizeHeap()` #11

**Bug fixes:**
 - build warning for cmake #33

**Misc:**
 - verification code and examples #35
 - install routines #4
 - See the full changes at https://github.com/ComputationalRadiationPhysics/mallocMC/compare/1.0.2crp...2.0.0crp


1.0.2crp
-------------
**Date:** 2014-01-07

This is our first bug fix release.
We closed all issues documented in
[Milestone *Bug fixes*](https://github.com/ComputationalRadiationPhysics/mallocMC/issues?milestone=1&state=closed)

### Changes to 1.0.1

**Features:**
  - added travis-ci.org support for compile tests #7

**Bug fixes:**
  - broken cmake/compile #1
  - g++ warnings #10
  - only N-1 access blocks used instead of N #2
  - 32bit bug: allocate more than 4GB #12

**Misc:**
  See the full changes at
  https://github.com/ComputationalRadiationPhysics/scatteralloc/compare/1.0.1...1.0.2crp
