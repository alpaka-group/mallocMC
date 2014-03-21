#include "src/include/scatteralloc/utils.h"
namespace GPUTools{

  template<bool b = true>
    class XMallocDistribution
    {
      bool can_use_coalescing;
      uint32 warpid;
      uint32 myoffset;
      uint32 threadcount;
      uint32 req_size;
      //const static uint32 pagesize = GetProperties<XMallocDistribution<b> >::pagesize;
      const static uint32 pagesize = GetProperties<XMallocDistribution<b> >::value::pagesize;


      public:
        __device__ uint32 gather(uint32 bytes){
          can_use_coalescing = false;
          warpid = GPUTools::warpid();
          myoffset = 0;
          threadcount = 0;
        
          //init with initial counter
          __shared__ uint32 warp_sizecounter[32];
          warp_sizecounter[warpid] = 16;

          bool coalescible = bytes > 0 && bytes < (pagesize / 32);
          uint32 threadcount = __popc(__ballot(coalescible));

          if (coalescible && threadcount > 1) 
          {
            myoffset = atomicAdd(&warp_sizecounter[warpid], bytes);
            can_use_coalescing = true;
          }

          req_size = bytes;
          if (can_use_coalescing)
            req_size = (myoffset == 16) ? warp_sizecounter[warpid] : 0;

          return req_size;
        }

        __device__ void* distribute(void* allocatedMem){
          __shared__ char* warp_res[32];

          char* myalloc = (char*) allocatedMem;
          if (req_size && can_use_coalescing) 
          {
            warp_res[warpid] = myalloc;
            if (myalloc != 0)
              *(uint32*)myalloc = threadcount;
          }
          __threadfence_block();

          void *myres = myalloc;
          if(can_use_coalescing) 
          {
            if(warp_res[warpid] != 0)
              myres = warp_res[warpid] + myoffset;
            else 
              myres = 0;
          }
          return myres;

        }

    };
}
