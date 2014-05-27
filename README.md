mallocMC
=============

mallocMC: Memory Allocator for Many Core Architectures

This project provides a framework for **fast memory managers** on **many core
accelerators**. Currently, it supports **NVIDIA GPUs** of compute capability
`sm_20` or higher through the ScatterAlloc algorithm.

From http://www.icg.tugraz.at/project/mvp/downloads :
```quote
ScatterAlloc is a dynamic memory allocator for the GPU. It is
designed concerning the requirements of massively parallel
execution.

ScatterAlloc greatly reduces collisions and congestion by
scattering memory requests based on hashing. It can deal with
thousands of GPU-threads concurrently allocating memory and its
execution time is almost independent of the thread count.

ScatterAlloc is open source and easy to use in your CUDA projects.
```

Original Homepage: http://www.icg.tugraz.at/project/mvp

Our Homepage: https://www.hzdr.de/crp


About This Repository
---------------------

The currently implemented algorithm is a
[fork](https://en.wikipedia.org/wiki/Fork_%28software_development%29)
of the **ScatterAlloc** project from the
[Managed Volume Processing](http://www.icg.tugraz.at/project/mvp)
group at [Institute for Computer Graphics and Vision](http://www.icg.tugraz.at),
TU Graz (kudos!).

Our aim is to improve the implementation, add new features and to fix some
minor bugs.


Branches
--------

| *branch*    | *state* | *description*           |
| ----------- | ------- | ----------------------- |
| **master**  | [![Build Status Master](https://travis-ci.org/ComputationalRadiationPhysics/scatteralloc.png?branch=master)](https://travis-ci.org/ComputationalRadiationPhysics/scatteralloc "master") | our stable new releases |
| **dev**     | [![Build Status Development](https://travis-ci.org/ComputationalRadiationPhysics/scatteralloc.png?branch=dev)](https://travis-ci.org/ComputationalRadiationPhysics/scatteralloc "dev") | our development branch - start and merge new branches here |
| **tugraz**  | n/a | kind-of the "upstream" branch - only used to receive new releases from the TU Graz group |


Install
-------

Installation notes can be found in [INSTALL.md](INSTALL.md).


Literature
----------

Just an incomplete link collection for now:

- [Paper](http://www.icg.tugraz.at/Members/steinber/scatteralloc-1) by
  Markus Steinberger, Michael Kenzel, Bernhard Kainz and Dieter Schmalstieg

- 2012, May 5th: [Presentation](http://innovativeparallel.org/Presentations/inPar_kainz.pdf)
        at *Innovative Parallel Computing 2012* by *Bernhard Kainz*


License
-------

We distribute the modified software under the same license as the
original software from TU Graz (by using the
[MIT License](https://en.wikipedia.org/wiki/MIT_License)).
Please refer to the [LICENSE](LICENSE) file.
