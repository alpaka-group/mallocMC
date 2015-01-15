Change Log / Release Log for mallocMC
================================================================

2.0.1crp
-------------
**Date:** 2015-01-13

This release fixes several bugs that occured after the release of 2.0.0crp.
We closed all issues documented in
[Milestone *Bugfixes*](https://github.com/ComputationalRadiationPhysics/mallocMC/issues?milestone=4&state=closed)

### Changes to mallocMC 2.0.0crp

**Bug fixes**
 - page table metadata was not correctly initialized with 0 #70
 - freeing pages would not work under certain circumstances #66
 - the bitmask in a page table entry could be wrong due to a racecondition #62
 - not all regions were initialized correctly #60
 - getAvailableSlots could sometimes miss blocks #59
 - the counter for elements in a page could get too high due to a racecondition #61
 - Out of Memory (OOM) Policy sometimes did not recognize allocation failures correctly #67

**Misc:**
 - See the full changes at https://github.com/ComputationalRadiationPhysics/mallocMC/compare/2.0.0crp...2.0.1crp


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
