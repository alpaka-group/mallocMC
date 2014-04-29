#pragma once

#include <stdio.h>
#include <boost/cstdint.hpp> /* uint32_t */
#include <iostream>
#include <string>

#include "../policy_malloc_utils.hpp"
#include "Scatter.hpp"

namespace PolicyMalloc{
namespace CreationPolicies{
    
namespace ScatterKernelDetail{
  template < typename T_Allocator >
  __global__ void initKernel(T_Allocator* heap, void* heapmem, size_t memsize){
    heap->initDeviceFunction(heapmem, memsize);
  }

  template < typename T_Allocator >
  __global__ void getAvailableSlotsKernel(T_Allocator* heap, void* pool, size_t slotSize, unsigned* slots){
    unsigned temp = heap->getAvailaibleSlotsDeviceFunction(pool, slotSize);
    if(temp) atomicAdd(slots, temp);
  }
}

  template<class T_Config, class T_Hashing>
  class Scatter
  {

    public:
      typedef T_Config  HeapProperties;
      typedef T_Hashing HashingProperties;
      struct  Properties : HeapProperties, HashingProperties{};
      
    private:
      typedef boost::uint32_t uint32;


#ifndef POLICYMALLOC_CP_SCATTER_PAGESIZE
#define POLICYMALLOC_CP_SCATTER_PAGESIZE  static_cast<uint32>(HeapProperties::pagesize::value)
#endif
      static const uint32 pagesize      = POLICYMALLOC_CP_SCATTER_PAGESIZE;

#ifndef POLICYMALLOC_CP_SCATTER_ACCESSBLOCKS
#define POLICYMALLOC_CP_SCATTER_ACCESSBLOCKS static_cast<uint32>(HeapProperties::accessblocks::value)
#endif
      static const uint32 accessblocks  = POLICYMALLOC_CP_SCATTER_ACCESSBLOCKS;

#ifndef POLICYMALLOC_CP_SCATTER_REGIONSIZE
#define POLICYMALLOC_CP_SCATTER_REGIONSIZE static_cast<uint32>(HeapProperties::regionsize::value)
#endif
      static const uint32 regionsize    = POLICYMALLOC_CP_SCATTER_REGIONSIZE;

#ifndef POLICYMALLOC_CP_SCATTER_WASTEFACTOR
#define POLICYMALLOC_CP_SCATTER_WASTEFACTOR static_cast<uint32>(HeapProperties::wastefactor::value)
#endif
      static const uint32 wastefactor   = POLICYMALLOC_CP_SCATTER_WASTEFACTOR;

#ifndef POLICYMALLOC_CP_SCATTER_RESETFREEDPAGES
#define POLICYMALLOC_CP_SCATTER_RESETFREEDPAGES static_cast<bool>(HeapProperties::resetfreedpages::value)
#endif
      static const bool resetfreedpages = POLICYMALLOC_CP_SCATTER_RESETFREEDPAGES;


    public:
      static const uint32 _pagesize       = pagesize;
      static const uint32 _accessblocks   = accessblocks;
      static const uint32 _regionsize     = regionsize;
      static const uint32 _wastefactor    = wastefactor;
      static const bool _resetfreedpages  = resetfreedpages;

    private:
#if _DEBUG || ANALYSEHEAP
    public:
#endif
      //static const uint32 minChunkSize0 = pagesize/(32*32);
      static const uint32 minChunkSize1 = 0x10;
      static const uint32 HierarchyThreshold =  (pagesize - 2*sizeof(uint32))/33;

#ifndef POLICYMALLOC_CP_SCATTER_HASHINGK
#define POLICYMALLOC_CP_SCATTER_HASHINGK    static_cast<uint32>(HashingProperties::hashingK::value)
#endif
     static const uint32 hashingK       = POLICYMALLOC_CP_SCATTER_HASHINGK;

#ifndef POLICYMALLOC_CP_SCATTER_HASHINGDISTMP
#define POLICYMALLOC_CP_SCATTER_HASHINGDISTMP static_cast<uint32>(HashingProperties::hashingDistMP::value)
#endif
     static const uint32 hashingDistMP  = POLICYMALLOC_CP_SCATTER_HASHINGDISTMP;

#ifndef POLICYMALLOC_CP_SCATTER_HASHINGDISTWP
#define POLICYMALLOC_CP_SCATTER_HASHINGDISTWP static_cast<uint32>(HashingProperties::hashingDistWP::value)
#endif
     static const uint32 hashingDistWP  = POLICYMALLOC_CP_SCATTER_HASHINGDISTWP;

#ifndef POLICYMALLOC_CP_SCATTER_HASHINGDISTWPREL
#define POLICYMALLOC_CP_SCATTER_HASHINGDISTWPREL static_cast<uint32>(HashingProperties::hashingDistWPRel::value)
#endif
     static const uint32 hashingDistWPRel = POLICYMALLOC_CP_SCATTER_HASHINGDISTWPREL;


      /**
       * Page Table Entry struct
       * The PTE holds basic information about each page
       */
      struct PTE
      {
        uint32 chunksize;
        uint32 count;
        uint32 bitmask;

        __device__ void init()
        {
          chunksize = 0;
          count = 0;
          bitmask = 0;
        }
      };

      /**
       * Page struct
       * The page struct is used to access the data on the page more efficiently
       * and to clear the area on the page, which might hold bitsfields later one
       */
      struct PAGE
      {
        char data[pagesize];

        /**
         * The pages init method
         * This method initializes the region on the page which might hold
         * bit fields when the page is used for a small chunk size
         * @param previous_chunksize the chunksize which was uses for the page before
         */
        __device__ void init(uint32 previous_chunksize = 0)
        {
          //TODO: we can speed this up for pages being freed, because we know the
          //chunksize used before (these bits must be zero again) 

          //init the entire data which can hold bitfields 
          uint32 max_bits = min(32*32,pagesize/minChunkSize1);
          uint32 max_entries = divup<uint32>(max_bits/8,sizeof(uint32))*sizeof(uint32);
          uint32* write = (uint32*)(data+(pagesize-max_entries));
          while(write < (uint32*)(data + pagesize))
            *write++ = 0;
        }
      };

      // the data used by the allocator

      volatile PTE* _ptes;
      volatile uint32* _regions;
      PAGE* _page;
      uint32 _numpages;
      size_t _memsize;
      uint32 _pagebasedMutex;
      volatile uint32 _firstFreePageBased;
      volatile uint32 _firstfreeblock;


      /**
       * randInit should create an random offset which can be used
       * as the initial position in a bitfield
       */
      __device__ inline uint32 randInit()
      {
        //start with the laneid offset
        return laneid();
      }

      /**
       * randInextspot delivers the next free spot in a bitfield
       * it searches for the next unset bit to the left of spot and
       * returns its offset. if there are no unset bits to the left
       * then it wraps around
       * @param bitfield the bitfield to be searched for
       * @param spot the spot from which to search to the left
       * @param spots number of bits that can be used
       * @return next free spot in the bitfield
       */
      __device__ inline uint32 nextspot(uint32 bitfield, uint32 spot, uint32 spots)
      {
        //wrap around the bitfields from the current spot to the left
        bitfield = ((bitfield >> (spot + 1)) | (bitfield << (spots - (spot + 1))))&((1<<spots)-1);
        //compute the step from the current spot in the bitfield
        uint32 step = __ffs(~bitfield);
        //and return the new spot
        return (spot + step) % spots;
      }

      /**
       * usespot marks finds one free spot in the bitfield, marks it and returns its offset
       * @param bitfield pointer to the bitfield to use
       * @param spots overall number of spots the bitfield is responsible for
       * @return if there is a free spot it returns the spot'S offset, otherwise -1
       */
      __device__ inline int usespot(uint32 *bitfield, uint32 spots)
      {
        //get first spot
        uint32 spot = randInit() % spots;
        for(;;)
        {
          uint32 mask = 1 << spot;
          uint32 old = atomicOr(bitfield, mask);
          if( (old & mask) == 0)
            return spot;
          // note: __popc(old) == spots should be sufficient, 
          //but if someone corrupts the memory we end up in an endless loop in here...
          if(__popc(old) >= spots)
            return -1;
          spot = nextspot(old, spot, spots);
        }
      }

      /**
       * addChunkHierarchy finds a free chunk on a page which uses bit fields on the page
       * @param chunksize the chunksize of the page
       * @param fullsegments the number of full segments on the page (a 32 bits on the page)
       * @param additional_chunks the number of additional chunks in last segment (less than 32 bits on the page)
       * @param page the page to use
       * @return pointer to a free chunk on the page, 0 if we were unable to obtain a free chunk
       */
      __device__ inline void* addChunkHierarchy(uint32 chunksize, uint32 fullsegments, uint32 additional_chunks, uint32 page)
      {
        uint32 segments = fullsegments + (additional_chunks > 0 ? 1 : 0);
        uint32 spot = randInit() % segments;
        uint32 mask = _ptes[page].bitmask;
        if((mask & (1 << spot)) != 0)
          spot = nextspot(mask, spot, segments);
        uint32 tries = segments - __popc(mask);
        uint32* onpagemasks = (uint32*)(_page[page].data + chunksize*(fullsegments*32 + additional_chunks));
        for(uint32 i = 0; i < tries; ++i)
        {
          int hspot = usespot(onpagemasks + spot, spot < fullsegments ? 32 : additional_chunks);
          if(hspot != -1)
            return _page[page].data + (32*spot + hspot)*chunksize;
          else
            atomicOr((uint32*)&_ptes[page].bitmask, 1 << spot);
          spot = nextspot(mask, spot, segments);
        }
        return 0;
      }

      /**
       * addChunkNoHierarchy finds a free chunk on a page which uses the bit fields of the pte only
       * @param chunksize the chunksize of the page
       * @param page the page to use
       * @param spots the number of chunks which fit on the page
       * @return pointer to a free chunk on the page, 0 if we were unable to obtain a free chunk
       */
      __device__ inline void* addChunkNoHierarchy(uint32 chunksize, uint32 page, uint32 spots)
      {
        int spot = usespot((uint32*)&_ptes[page].bitmask, spots);
        if(spot == -1)
          return 0; //that should be impossible :)
        return _page[page].data + spot*chunksize;
      }

      /**
       * tryUsePage tries to use the page for the allocation request
       * @param page the page to use
       * @param chunksize the chunksize of the page
       * @return pointer to a free chunk on the page, 0 if we were unable to obtain a free chunk
       */
      __device__ inline void* tryUsePage(uint32 page, uint32 chunksize)
      {
        //increse the fill level
        uint32 filllevel = atomicAdd((uint32*)&(_ptes[page].count), 1);
        //recheck chunck size (it could be that the page got freed in the meanwhile...)
        if(!resetfreedpages || _ptes[page].chunksize == chunksize)
        {
          if(chunksize <= HierarchyThreshold)
          {
            //more chunks than can be covered by the pte's single bitfield can be used
            uint32 segmentsize = chunksize*32 + sizeof(uint32);
            uint32 fullsegments = 0;
            uint32 additional_chunks = 0;
            fullsegments = pagesize / segmentsize;
            additional_chunks = max(0,(int)pagesize - (int)fullsegments*segmentsize - (int)sizeof(uint32))/chunksize;
            if(filllevel < fullsegments * 32 + additional_chunks)
              return addChunkHierarchy(chunksize, fullsegments, additional_chunks, page);
          }
          else
          {
            uint32 chunksinpage = min(pagesize / chunksize, 32);
            if(filllevel < chunksinpage)
              return addChunkNoHierarchy(chunksize, page, chunksinpage);
          }
        }

        //this one is full/not useable
        atomicSub((uint32*)&(_ptes[page].count), 1);
        return 0;
      }

      /**
       * allocChunked tries to allocate the demanded number of bytes on one of the pages
       * @param bytes the number of bytes to allocate
       * @return pointer to a free chunk on a page, 0 if we were unable to obtain a free chunk
       */
      __device__ void* allocChunked(uint32 bytes)
      {
        uint32 pagesperblock = _numpages/accessblocks;
        uint32 reloff = warpSize*bytes / pagesize;
        uint32 startpage = (bytes*hashingK + hashingDistMP*smid() + (hashingDistWP+hashingDistWPRel*reloff)*warpid() ) % pagesperblock;
        uint32 maxchunksize = min(pagesize,wastefactor*bytes);
        uint32 startblock = _firstfreeblock;
        uint32 ptetry = startpage + startblock*pagesperblock;
        uint32 checklevel = regionsize*3/4;
        for(uint32 finder = 0; finder < 2; ++finder)
        {
          for(uint32 b = startblock; b < accessblocks; ++b)
          {
            while(ptetry < (b+1)*pagesperblock)
            {
              uint32 region = ptetry/regionsize;
              uint32 regionfilllevel = _regions[region];
              if(regionfilllevel < checklevel )
              {
                for( ; ptetry < (region+1)*regionsize; ++ptetry)
                {
                  uint32 chunksize = _ptes[ptetry].chunksize;
                  if(chunksize >= bytes && chunksize <= maxchunksize)
                  {            
                    void * res = tryUsePage(ptetry, chunksize);
                    if(res != 0)  return res;
                  }
                  else if(chunksize == 0)
                  {
                    //lets open up a new page
                    //it is already padded
                    uint32 new_chunksize = max(bytes,minChunkSize1);
                    uint32 beforechunksize = atomicCAS((uint32*)&_ptes[ptetry].chunksize, 0, new_chunksize);
                    if(beforechunksize == 0)
                    {
                      void * res = tryUsePage(ptetry, new_chunksize);
                      if(res != 0)  return res;
                    }
                    else if(beforechunksize >= bytes &&  beforechunksize <= maxchunksize)
                    {
                      //someone else aquired the page, but we can also use it
                      void * res = tryUsePage(ptetry, beforechunksize);
                      if(res != 0)  return res;
                    }
                  }
                }
                //could not alloc in region, tell that
                if(regionfilllevel + 1 <= regionsize)
                  atomicMax((uint32*)(_regions + region), regionfilllevel+1);
              }
              else
                ptetry += regionsize;
              //ptetry = (region+1)*regionsize;
            }
            //randomize the thread writing the info
            //if(warpid() + laneid() == 0)
            if(b > startblock)
              _firstfreeblock = b;
          }

          //we are really full :/ so lets search every page for a spot!
          startblock = 0;
          checklevel = regionsize + 1;
          ptetry = 0;
        }
        return 0;
      }
      
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      __device__ unsigned countFreeChunksInPage(uint32 currentpage, uint32 chunksize){
          
        return 0;
      }



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      __device__ unsigned getAvailaibleSlotsDeviceFunction(void* pool, size_t slotSize)
      {
        if(slotSize < pagesize) // multiple slots per page
        {

          uint32 pagespersuperblock = _numpages/accessblocks;
          uint32 reloff = warpSize*bytes / pagesize;
          //uint32 startpage = (bytes*hashingK + hashingDistMP*smid() + (hashingDistWP+hashingDistWPRel*reloff)*warpid() ) % pagespersuperblock;
          uint32 startpage = 0;
          uint32 maxchunksize = min(pagesize,wastefactor*bytes);
          //uint32 startblock = _firstfreeblock;
          //uint32 currentpage = startpage + startblock*pagespersuperblock; //optimization: check only non-full superblocks
          uint32 checklevel = regionsize*3/4; // this can be used to adjust precision (set to *1 to get maximum precision)
          unsigned slotcount = 0;

          for(uint32 currentpage=0; currentpage < numpages; ++currentpage){ //this already includes the superblocks
            uint32 region = currentpage/regionsize;
            uint32 regionfilllevel = _regions[region];
            if(regionfilllevel < checklevel){
              uint32 chunksize = _ptes[currentpage].chunksize;
              if(chunksize >= bytes && chunksize <= maxchunksize){
                //see how much space is left (each chunk suffices to satisfy our request)
                slotcount += countFreeChunksInPage(currentpage, chunksize);
              }else if(chunksize==0){
                //take it all
                chunksize = max(bytes, minChunkSize1); //make sure the chunks are big enough for the request the heap limits
                slotcount += countFreeChunksInPage(currentpage, chunksize);
              }else{
                 //do nothing: the chunks here are too small for the request :( 
                 continue;
              }// chunksize >= bytes
            }//if(regionfilllevel...)
          }//currentpage
        }else // 1 slot needs multiple pages
        {
          uint32 pagestoalloc = divup(bytes, pagesize);
          uint32 freecount = 0;
          unsigned slotcount=0;
          for(uint32 currentpage=0; currentpage < numpages; ++currentpage){ //this already includes the superblocks
            if(_ptes[currentpage].chunksize == 0){
              if(++freecount == pagestoalloc){
                freecount = 0;
                ++slotcount; //TODO: use this slotcount in final result
              }
            }
          }
        }
        return slotcount;
      }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


      /**
       * deallocChunked frees the chunk on the page and updates all data accordingly
       * @param mem pointer to the chunk
       * @param page the page the chunk is on
       * @param chunksize the chunksize used for the page
       */
      __device__ void deallocChunked(void* mem, uint32 page, uint32 chunksize)
      {
        uint32 inpage_offset = ((char*)mem - _page[page].data);
        if(chunksize <= HierarchyThreshold)
        {
          //one more level in hierarchy
          uint32 segmentsize = chunksize*32 + sizeof(uint32);
          uint32 fullsegments = pagesize / segmentsize;
          uint32 additional_chunks = max(0,(int)(pagesize - fullsegments*segmentsize) - (int)sizeof(uint32))/chunksize;
          uint32 segment = inpage_offset / (chunksize*32);
          uint32 withinsegment = (inpage_offset - segment*(chunksize*32))/chunksize;
          //mark it as free
          uint32* onpagemasks = (uint32*)(_page[page].data + chunksize*(fullsegments*32 + additional_chunks));
          uint32 old = atomicAnd(onpagemasks + segment, ~(1 << withinsegment));

          uint32 elementsinsegment = segment < fullsegments ? 32 : additional_chunks;
          if(__popc(old) == elementsinsegment)
            atomicAnd((uint32*)&_ptes[page].bitmask, ~(1 << segment));
        }
        else
        {
          uint32 segment = inpage_offset / chunksize;
          atomicAnd((uint32*)&_ptes[page].bitmask, ~(1 << segment));
        }
        //reduce filllevel as free
        uint32 oldfilllevel = atomicSub((uint32*)&_ptes[page].count, 1);


        if(resetfreedpages)
        {
          if(oldfilllevel == 1)
          {
            //this page now got free!
            // -> try lock it
            uint32 old = atomicCAS((uint32*)&_ptes[page].count, 0, pagesize);
            if(old == 0)
            {
              //clean the bits for the hierarchy
              _page[page].init();
              //remove chunk information
              _ptes[page].chunksize = 0;
              __threadfence();
              //unlock it
              atomicSub((uint32*)&_ptes[page].count, pagesize);
            }
          }
        }

        //meta information counters ... should not be changed by too many threads, so..
        if(oldfilllevel == pagesize / 2 / chunksize)
        {
          uint32 region = page / regionsize;
          _regions[region] = 0;        
          uint32 block = region * regionsize * accessblocks / _numpages ;
          if(warpid() + laneid() == 0)
            atomicMin((uint32*)&_firstfreeblock, block);
        }
      }

      /**
       * markpages markes a fixed number of pages as used
       * @param startpage first page to mark
       * @param pages number of pages to mark
       * @param bytes number of overall bytes to mark pages for
       * @return true on success, false if one of the pages is not free
       */
      __device__ bool markpages(uint32 startpage, uint32 pages, uint32 bytes)
      {
        int abord = -1;
        for(uint32 trypage = startpage; trypage < startpage + pages; ++trypage)
        {
          uint32 old = atomicCAS((uint32*)&_ptes[trypage].chunksize, 0, bytes);
          if(old != 0)
          {
            abord = trypage;
            break;
          }
        }
        if(abord == -1)
          return true;
        for(uint32 trypage = startpage; trypage < abord; ++trypage)
          atomicCAS((uint32*)&_ptes[trypage].chunksize, bytes, 0);
        return false;
      }

      /**
       * allocPageBasedSingleRegion tries to allocate the demanded number of bytes on a continues sequence of pages
       * @param startpage first page to be used
       * @param endpage last page to be used
       * @param bytes number of overall bytes to mark pages for
       * @return pointer to the first page to use, 0 if we were unable to use all the requested pages
       */
      __device__ void* allocPageBasedSingleRegion(uint32 startpage, uint32 endpage, uint32 bytes)
      {
        uint32 pagestoalloc = divup(bytes, pagesize);
        uint32 freecount = 0;
        bool left_free = false;
        for(uint32 search_page = startpage+1; search_page > endpage; )
        {
          --search_page;
          if(_ptes[search_page].chunksize == 0)
          {
            if(++freecount == pagestoalloc)
            {
              //try filling it up
              if(markpages(search_page, pagestoalloc, bytes))
              {
                //mark that we filled up everything up to here
                if(!left_free)
                  atomicCAS((uint32*)&_firstFreePageBased, startpage, search_page - 1);
                return _page[search_page].data;
              }
            }
          }
          else
          {
            left_free = true;
            freecount = 0;
          }
        }
        return 0;
      }

      /**
       * allocPageBasedSingle tries to allocate the demanded number of bytes on a continues sequence of pages
       * @param bytes number of overall bytes to mark pages for
       * @return pointer to the first page to use, 0 if we were unable to use all the requested pages
       * @pre only a single thread of a warp is allowed to call the function concurrently
       */
      __device__ void* allocPageBasedSingle(uint32 bytes)
      {
        //acquire mutex
        while(atomicExch(&_pagebasedMutex,1) != 0);
        //search for free spot from the back
        uint32 spage = _firstFreePageBased;
        void* res = allocPageBasedSingleRegion(spage, 0, bytes);
        if(res == 0)
          //also check the rest of the pages
          res = allocPageBasedSingleRegion(_numpages, spage, bytes);

        //free mutex
        atomicExch(&_pagebasedMutex,0);
        return res;
      }
      /**
       * allocPageBased tries to allocate the demanded number of bytes on a continues sequence of pages
       * @param bytes number of overall bytes to mark pages for
       * @return pointer to the first page to use, 0 if we were unable to use all the requested pages
       */
      __device__ void* allocPageBased(uint32 bytes)
      {
        //this is rather slow, but we dont expect that to happen often anyway

        //only one thread per warp can acquire the mutex
        void* res = 0;
        warp_serial
          res = allocPageBasedSingle(bytes);
        return res;
      }

      /**
       * deallocPageBased frees the memory placed on a sequence of pages
       * @param mem pointer to the first page
       * @param page the first page
       * @param bytes the number of bytes to be freed
       */
      __device__ void deallocPageBased(void* mem, uint32 page, uint32 bytes)
      {
        uint32 pages = divup(bytes,pagesize);
        for(uint32 p = page; p < page+pages; ++p)
          _page[p].init();
        __threadfence();
        for(uint32 p = page; p < page+pages; ++p)
          atomicCAS((uint32*)&_ptes[p].chunksize, bytes, 0);
        atomicMax((uint32*)&_firstFreePageBased, page+pages-1);
      }


    public:
      /**
       * create allocates the requested number of bytes via the heap. Coalescing has to be done before by another policy.
       * @param bytes number of bytes to allocate
       * @return pointer to the allocated memory
       */
      __device__ void* create(uint32 bytes)
      {
        if(bytes == 0)
          return 0;
        //take care of padding
        //bytes = (bytes + dataAlignment - 1) & ~(dataAlignment-1); // in alignment-policy
        if(bytes < pagesize)
          //chunck based 
          return allocChunked(bytes);
        else
          //allocate a range of pages
          return allocPageBased(bytes);
      }

      /**
       * destroy frees the memory regions previously acllocted via create
       * @param mempointer to the memory region to free
       */
      __device__ void destroy(void* mem)
      {
        if(mem == 0)
          return;
        //lets see on which page we are on
        uint32 page = ((char*)mem - (char*)_page)/pagesize;
        uint32 chunksize = _ptes[page].chunksize;

        //is the pointer the beginning of a chunk?
        uint32 inpage_offset = ((char*)mem - _page[page].data);
        uint32 block = inpage_offset/chunksize;
        uint32 inblockoffset = inpage_offset - block*chunksize;
        if(inblockoffset != 0)
        {
          uint32* counter = (uint32*)(_page[page].data + block*chunksize);
          //coalesced mem free
          uint32 old = atomicSub(counter, 1);
          if(old != 1)
            return;
          mem = (void*) counter;
        }

        if(chunksize < pagesize)
          deallocChunked(mem, page, chunksize);
        else
          deallocPageBased(mem, page, chunksize);
      }

      /**
       * init inits the heap data structures
       * the init method must be called before the heap can be used. the method can be called
       * with an arbitrary number of threads, which will increase the inits efficiency
       * @param memory pointer to the memory used for the heap
       * @param memsize size of the memory in bytes
       */
      __device__ void initDeviceFunction(void* memory, size_t memsize)
      {
        uint32 linid = threadIdx.x + blockDim.x*(threadIdx.y + threadIdx.z*blockDim.y);
        uint32 threads = blockDim.x*blockDim.y*blockDim.z;
        uint32 linblockid = blockIdx.x + gridDim.x*(blockIdx.y + blockIdx.z*gridDim.y);
        uint32 blocks =  gridDim.x*gridDim.y*gridDim.z;
        linid = linid + linblockid*threads;

        uint32 numregions = ((unsigned long long)memsize)/( ((unsigned long long)regionsize)*(sizeof(PTE)+pagesize)+sizeof(uint32));
        uint32 numpages = numregions*regionsize;
        //pointer is copied (copy is called page)
        PAGE* page = (PAGE*)(memory);
        //sec check for alignment
        //copy is checked
        //PointerEquivalent alignmentstatus = ((PointerEquivalent)page) & (16 -1);
        //if(alignmentstatus != 0)
        //{
        //  if(linid == 0){
        //    printf("c Before:\n");
        //    printf("c dataAlignment:   %d\n",16);
        //    printf("c Alignmentstatus: %d\n",alignmentstatus);
        //    printf("c size_t memsize   %llu byte\n", memsize);
        //    printf("c void *memory     %p\n", page);
        //  }
        //  //copy is adjusted, potentially pointer to higher address now.
        //  page =(PAGE*)(((PointerEquivalent)page) + 16 - alignmentstatus);
        //  if(linid == 0) printf("c Heap Warning: memory to use not 16 byte aligned...\n");
        //}
        PTE* ptes = (PTE*)(page + numpages);
        uint32* regions = (uint32*)(ptes + numpages);
        //sec check for mem size
        //this check refers to the original memory-pointer, which was not adjusted!
        if( (void*)(regions + numregions) > (((char*)memory) + memsize) )
        {
          --numregions;
          numpages = min(numregions*regionsize,numpages);
          if(linid == 0) printf("c Heap Warning: needed to reduce number of regions to stay within memory limit\n");
        }
        //if(linid == 0) printf("Heap info: wasting %d bytes\n",(((POINTEREQUIVALENT)memory) + memsize) - (POINTEREQUIVALENT)(regions + numregions));

        //if(linid == 0 && alignmentstatus != 0){
        //  printf("c Was shrinked automatically to:\n");
        //  printf("c size_t memsize   %llu byte\n", memsize);
        //  printf("c void *memory     %p\n", page);
        //}
        threads = threads*blocks;

        for(uint32 i = linid; i < numpages; i+= threads)
        {
          ptes[i].init();
          page[i].init();
        }
        for(uint32 i = linid; i < numregions; i+= numregions)
          regions[i] = 0;

        if(linid == 0)
        {
          _memsize = memsize;
          _numpages = numpages;
          _ptes = (volatile PTE*)ptes;
          _page = page;
          _regions =  regions;
          _firstfreeblock = 0;
          _pagebasedMutex = 0;
          _firstFreePageBased = numpages-1;

          if( (char*) (_page+numpages) > (char*)(memory) + memsize)
            printf("error in heap alloc: numpages too high\n");
        }

      }


      __device__ bool isOOM(void* p){
        // all threads in a warp return get NULL
        return  32 == __popc(__ballot(p == NULL));
      }


      template < typename T_Obj>
      static void* initHeap(const T_Obj& obj, void* pool, size_t memsize){
        T_Obj* heap;
        SCATTERALLOC_CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&heap,obj));
        ScatterKernelDetail::initKernel<<<1,256>>>(heap, pool, memsize);
        return heap;
      }


      template < typename T_Obj >
      static void finalizeHeap(const T_Obj& obj, void* pool){
        /* @TODO: Think about the necessity of a teardown... (inside the pool) */
        //reset PAGE, memsize, numpages, regions, firstfreedblock, firstfreepagebased,numregions,ptes
      }

      template <typename T_Obj>
      static unsigned getAvailableSlots(const T_Obj& obj, void* pool, size_t slotSize){
        T_Obj* heap;
        SCATTERALLOC_CUDA_CHECKED_CALL(cudaGetSymbolAddress((void**)&heap,obj));
        unsigned h_slots = 0;
        unsigned* d_slots;
        cudaMalloc((void**) &d_slots, sizeof(unsigned));
        cudaMemcpy(d_slots, &h_slots, sizeof(unsigned), cudaMemcpyHostToDevice);
      
        ScatterKernelDetail::getAvailableSlotsKernel<<<1,1>>>(heap, pool, slotSize, d_slots);

        cudaMemcpy(&h_slots, d_slots, sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaFree(d_slots);
        return h_slots;
      }

      static std::string classname(){
        std::stringstream ss;
        ss << "Scatter[";
        ss << pagesize        << ",";
        ss << accessblocks    << ",";
        ss << regionsize      << ",";
        ss << wastefactor     << ",";
        ss << resetfreedpages << ",";
        ss << hashingK        << ",";
        ss << hashingDistMP   << ",";
        ss << hashingDistWP   << ",";
        ss << hashingDistWPRel<< "]";
        return ss.str();
      }

  };

} //namespace CreationPolicies
} //namespace PolicyMalloc
