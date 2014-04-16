#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>

#include <cuda.h>
#include <boost/mpl/int.hpp>
#include <boost/mpl/bool.hpp>

///////////////////////////////////////////////////////////////////////////////
// includes for PolicyMalloc
///////////////////////////////////////////////////////////////////////////////
// basic files for PolicyMalloc
#include "src/include/scatteralloc/policy_malloc_overwrites.hpp"
#include "src/include/scatteralloc/policy_malloc_hostclass.hpp"

// Load all available policies for PolicyMalloc
#include "src/include/scatteralloc/CreationPolicies.hpp"
#include "src/include/scatteralloc/DistributionPolicies.hpp"
#include "src/include/scatteralloc/OOMPolicies.hpp"
#include "src/include/scatteralloc/GetHeapPolicies.hpp"
#include "src/include/scatteralloc/AlignmentPolicies.hpp"
    
///////////////////////////////////////////////////////////////////////////////
// Configuration for PolicyMalloc
///////////////////////////////////////////////////////////////////////////////

// configurate the CreationPolicy "Scatter"
struct ScatterConfig{
    typedef boost::mpl::int_<4096>  pagesize;
    typedef boost::mpl::int_<8>     accessblocks;
    typedef boost::mpl::int_<16>    regionsize;
    typedef boost::mpl::int_<2>     wastefactor;
    typedef boost::mpl::bool_<false> resetfreedpages;
};

struct ScatterHashParams{
    typedef boost::mpl::int_<38183> hashingK;
    typedef boost::mpl::int_<17497> hashingDistMP;
    typedef boost::mpl::int_<1>     hashingDistWP;
    typedef boost::mpl::int_<1>     hashingDistWPRel;
};

// configure the DistributionPolicy "XMallocSIMD"
struct DistributionConfig{
  typedef ScatterConfig::pagesize pagesize;
};

// configure the AlignmentPolicy "Shrink"
struct AlignmentConfig{
  typedef boost::mpl::int_<16> dataAlignment;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behaviour of ScatterAlloc
typedef PolicyMalloc::PolicyAllocator< 
  PolicyMalloc::CreationPolicies::Scatter<ScatterConfig,ScatterHashParams>,
  PolicyMalloc::DistributionPolicies::XMallocSIMD<DistributionConfig>,
  PolicyMalloc::OOMPolicies::ReturnNull,
  PolicyMalloc::GetHeapPolicies::SimpleCudaMalloc,
  PolicyMalloc::AlignmentPolicies::Shrink<AlignmentConfig>
  > ScatterAllocator;

// use "ScatterAllocator" as PolicyAllocator
SET_ACCELERATOR_MEMORY_ALLOCATOR_TYPE(ScatterAllocator)

// replace all standard malloc()-calls on the device by PolicyAllocator calls
// This will not work with the CreationPolicy "OldMalloc"!
POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE()

///////////////////////////////////////////////////////////////////////////////
// End of PolicyMalloc configuration
///////////////////////////////////////////////////////////////////////////////


void run();

int main()
{
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  if( deviceProp.major < 2 ) {
    std::cerr << "Error: Compute Capability >= 2.0 required. (is ";
    std::cerr << deviceProp.major << "."<< deviceProp.minor << ")" << std::endl;
    return 1;
  }

  cudaSetDevice(0);
  run();
  cudaDeviceReset();

  return 0;
}


__device__ int** a;
__device__ int** b;
__device__ int** c;


__global__ void createArrays(int x, int y){
  a = (int**) malloc(sizeof(int*) * x*y); 
  b = (int**) malloc(sizeof(int*) * x*y); 
  c = (int**) malloc(sizeof(int*) * x*y); 
}


__global__ void fillArrays(int length, int* d){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  // using the POLICY_MALLOC_MEMORY_ALLOCATOR_MALLOC_OVERWRITE() macro
  // allows also the use of "new" 
  a[id] = new int[length];
  b[id] = new int[length];
  c[id] = new int[length];
  
  for(int i=0 ; i<length; ++i){
    a[id][i] = id*length+i; 
    b[id][i] = id*length+i;
  }
}


__global__ void addArrays(int length, int* d){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  d[id] = 0;
  for(int i=0 ; i<length; ++i){
    c[id][i] = a[id][i] + b[id][i];
    d[id] += c[id][i];
  }
}


__global__ void freeArrays(){
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  delete(a[id]);
  delete(b[id]);
  delete(c[id]);
}


void run()
{
  size_t block = 32;
  size_t grid = 32;
  int length = 100;
  assert(length<= block*grid); //necessary for used algorithm

  //init the heap
  std::cerr << "initHeap...";
  PolicyMalloc::initHeap(1U*1024U*1024U*1024U); //1GB for device-side malloc
  std::cerr << "done" << std::endl;

  // device-side pointers
  int*  d;
  cudaMalloc((void**) &d, sizeof(int)*block*grid);

  // host-side pointers
  std::vector<int> array_sums(block*grid,0);

  // create arrays of arrays on the device
  createArrays<<<1,1>>>(grid,block);

  // fill 2 of them all with ascending values
  fillArrays<<<grid,block>>>(length, d);

  // add the 2 arrays (vector addition within each thread)
  // and do a thread-wise reduce to d
  addArrays<<<grid,block>>>(length, d);

  cudaMemcpy(&array_sums[0],d,sizeof(int)*block*grid,cudaMemcpyDeviceToHost);

  int sum = std::accumulate(array_sums.begin(),array_sums.end(),0);
  std::cout << "The sum of the arrays on GPU is " << sum << std::endl;

  int n = block*grid*length;
  int gaussian = n*(n-1);
  std::cout << "The gaussian sum as comparison: " << gaussian << std::endl;

  freeArrays<<<grid,block>>>();
  cudaFree(d);
  //finalize the heap again
  PolicyMalloc::finalizeHeap();
}
