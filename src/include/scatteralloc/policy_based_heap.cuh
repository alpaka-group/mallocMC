#include "src/include/scatteralloc/utils.h"

namespace GPUTools{

  template < typename T_Allocator >
    __global__ void initKernel(T_Allocator* heap, void* heapmem, size_t memsize){
      heap->initDeviceFunction(heapmem, memsize);
    }


  template < 
     typename T_CreationPolicy, 
     typename T_AllocationPolicy, 
     typename T_OOMPolicy, 
     typename T_GetHeapPolicy
       >
  struct PolicyAllocator : 
    public T_CreationPolicy, 
    public T_AllocationPolicy, 
    public T_OOMPolicy, 
    public T_GetHeapPolicy
  {
    private:
      static const uint32 dataAlignment=0x10;
      typedef T_CreationPolicy CreationPolicy;
      typedef T_AllocationPolicy AllocationPolicy;
      typedef T_OOMPolicy OOMPolicy;
      typedef T_GetHeapPolicy GetHeapPolicy;

    public:
      typedef PolicyAllocator<CreationPolicy,AllocationPolicy,OOMPolicy,GetHeapPolicy> MyType;
      __device__ void* alloc(size_t bytes){
        AllocationPolicy allocationPolicy;

        bytes = (bytes + dataAlignment - 1) & ~(dataAlignment-1); // TODO: own policy? trait?

        uint32 req_size  = allocationPolicy.gather(bytes); //TODO: still needs pagesize
        void* memBlock   = CreationPolicy::create(req_size);
        const bool oom   = CreationPolicy::isOOM(memBlock);
        if(oom) memBlock = OOMPolicy::handleOOM(memBlock);
        void* myPart     = allocationPolicy.distribute(memBlock); //TODO: still needs pagesize

        return myPart;
        // if(blockIdx.x==0 && threadIdx.x==0){
        //     printf("warp %d trying to allocate %d bytes. myalloc: %p (oom %d)\n",GPUTools::warpid(),req_size,myalloc,oom);
        // }
      };

      __device__ void dealloc(void* p){
        CreationPolicy::destroy(p);
      };

      __host__ static void* initHeap(const MyType& obj, size_t size){
        void* pool = GetHeapPolicy::getMemPool(size);
        return CreationPolicy::initHeap(obj,pool,size);
      }

      __host__ static void destroyHeap(const MyType& obj){
        GetHeapPolicy::freeMemPool(obj);
      }

  };


}

