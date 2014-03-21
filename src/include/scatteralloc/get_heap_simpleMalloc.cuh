#include "src/include/scatteralloc/utils.h"
#include <assert.h>

namespace GPUTools{
  struct GetHeapSimpleMalloc
  {
      static void* getMemPool(size_t memsize){
        void* pool;
        SCATTERALLOC_CUDA_CHECKED_CALL(cudaMalloc(&pool, memsize));
        return pool;
      }

    template < typename T>
      static void freeMemPool(const T& obj){
        //assert(!"freeMemPool is not implemented!");
        //TODO implement me!
      }
  };
}
