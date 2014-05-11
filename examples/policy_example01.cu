#include <iostream>
#include <assert.h>
#include <vector>
#include <numeric>

#include <cuda.h>
#include "policy_example01_config.hpp"

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
  a = (int**) PolicyMalloc::malloc(sizeof(int*) * x*y); 
  b = (int**) PolicyMalloc::malloc(sizeof(int*) * x*y); 
  c = (int**) PolicyMalloc::malloc(sizeof(int*) * x*y); 
}


__global__ void fillArrays(int length, int* d){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  a[id] = (int*) PolicyMalloc::malloc(length*sizeof(int));
  b[id] = (int*) PolicyMalloc::malloc(length*sizeof(int));
  c[id] = (int*) PolicyMalloc::malloc(sizeof(int)*length);
  
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
  PolicyMalloc::free(a[id]);
  PolicyMalloc::free(b[id]);
  PolicyMalloc::free(c[id]);
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

  std::cout << ScatterAllocator::info("\n") << std::endl;

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

  PolicyMalloc::getAvailableSlotsHost(1024U*1024U); //get available megabyte-sized slots

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
