Usage
=====

Step 1: include
---------------

There is one header file that will include *all* necessary files:

```c++
#include <scatteralloc/scatteralloc.hpp>
```

Step 2a: choose policies
-----------------------

Each instance of a policy based allocator is composed through 5 **policies**. Each policy is expressed as a **policy class**. 

Currently, there are the following policy classes available:

|Policy                 | Policy Classes (implementations) | description |
|-------                |----------------------------------| ----------- |
|**CreationPolicy**     | Scatter`<conf1,conf2>`           | A scattered allocation to tradeoff fragmentation for allocation time, as proposed in [ScatterAlloc](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6339604). `conf1` configures the heap layout, `conf2` determines the hashing parameters|
|                       | OldMalloc                        | device-side malloc/new and free/delete syscalls as implemented on NVidia CUDA graphics cards with compute capability sm_20 and higher |
|**DistributionPolicy** | XMallocSIMD`<conf>`              | SIMD optimization for warp-wide allocation on NVIDIA CUDA accelerators, as proposed by [XMalloc](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5577907). `conf` is used to determine the pagesize. If used in combination with *Scatter*, the pagesizes must match |
|                       | Noop                             | no workload distribution at all |
|**OOMPolicy**          | ReturnNull                       | pointers will be *NULL*, if the request could not be fulfilled |
|                       | ~~BadAllocException~~            | will throw a `std::bad_alloc` exception. The accelerator has to support exceptions |
|**ReservePoolPolicy**  | SimpleCudaMalloc                 | allocate a fixed heap with `CudaMalloc` |
|                       | CudaSetLimits                    | call to `CudaSetLimits` to increase the available Heap (e.g. when using *OldMalloc*) |
|**AlignmentPolicy**    | Shrink`<conf>`                   | shrinks the pool so that the starting pointer is well aligned, applies padding to requested memory chunks. `conf` is used to determine the alignment|
|                       | Noop                             | no alignment at all |

The user has to choose one of each policy that will form a useful allocator
(see [here](Usage.md#2c-combine-policies))

Step 2b: configure policies
---------------------------

Some of those policies are templates that can be configured through a
configuration struct. The default struct can be accessed through
```PolicyNamespace::PolicyClass<>::Properties```, which allows to
inherit a struct to modify some of its parameters before passing it
to the policy class:

```c++
// configure the AlignmentPolicy "Shrink"
struct ShrinkConfig : PolicyMalloc::AlignmentPolicies::Shrink<>::Properties {
  typedef boost::mpl::int_<16> dataAlignment;
};
```

Step 2c: combine policies
-------------------------
After configuring the chosen policies, they can be used as template
parameters to create the desired allocator type:

```c++
using namespace PolicyMalloc;

typedef PolicyAllocator<
  CreationPolicy::OldMalloc,
  DistributionPolicy::Noop,
  OOMPolicy::ReturnNull,
  ReservePoolPolicy::CudaSetLimits,
  AlignmentPolicy::Noop
> Allocator1;
```

`Allocator1` will resemble the behaviour of classical device-side allocation known
from NVIDIA CUDA since compute capability sm_20. To get a more novel allocator, one
could create the following typedef instead:

```c++
using namespace PolicyMalloc;

typedef PolicyAllocator< 
  CreationPolicies::Scatter<>,
  DistributionPolicies::XMallocSIMD<>,
  OOMPolicies::ReturnNull,
  ReservePoolPolicies::SimpleCudaMalloc,
  AlignmentPolicies::Shrink<ShrinkConfig>
> ScatterAllocator;
```

Notice, how the policy classes `Scatter` and `XMallocSIMD` are instantiated without
template arguments to use the default configuration. `Shrink` however uses the
configuration struct defined above.


Step 3: instantiate allocator
-----------------------------

To create a default instance of the ScatterAllocator type and add the necessary 
functions, the following Macro has to be executed:

```c++
POLICYMALLOC_SET_ALLOCATOR_TYPE(ScatterAllocator)
```

This will set up the following functions in the namespace `PolicyMalloc`:

| Name                  | description                                                                                                                                                                                                |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| InitHeap()            | Initializes the heap. Must be called before any other calls to the allocator are permitted. Can take the desired size of the heap as a parameter                                                           |
| finalizeHeap()        | Destroys the heap again                                                                                                                                                                                    |
| pbMalloc() / malloc() | Allocates memory on the accelerator                                                                                                                                                                        |
| pbFree() / free()     | Frees memory on the accelerator                                                                                                                                                                            |
| getAvailableSlots()   | Determines number of allocatable slots of a certain size. This only works, if the chose CreationPolicy supports it (can be found through `PolicyMalloc::Traits<ScatterAllocator>::providesAvailableSlots`) |

If the policy class `OldMalloc` is **not** used, it is also possible to execute
the Macro
```c++
POLICYMALLOC_OVERWRITE_MALLOC()
```

which will overwrite the global functions `malloc()`/`free()` on the accelerator
(for NVIDIA CUDA accelerators, this will also replace calls to `new` and `delete`).


Step 4: use dynamic memory allocation
-------------------------------------
A simplistic example would look like this:
```c++
#include <scatteralloc/scatteralloc.hpp>

namespace PolicyMalloc = PM;

typedef PM::PolicyAllocator< 
  PM::CreationPolicies::Scatter<>,
  PM::DistributionPolicies::XMallocSIMD<>,
  PM::OOMPolicies::ReturnNull,
  PM::ReservePoolPolicies::SimpleCudaMalloc,
  PM::AlignmentPolicies::Shrink<ShrinkConfig>
  > ScatterAllocator;

POLICYMALLOC_SET_ALLOCATOR_TYPE(ScatterAllocator)

__global__ exampleKernel()
{
  // some code ...
  
  int* a = (int*) PM::malloc(sizeof(int)*42);
  
  // some more code, using *a
  
  PM::free(a);
}

int main(){
  PM::initHeap(512); // heapsize of 512MB

  exampleKernel<<<32,32>>>();

  PM::finalizeHeap();
  return 0;
}
```
