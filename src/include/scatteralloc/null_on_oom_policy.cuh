class NullOnOOM
{
  public:
    __device__ static void* handleOOM(void* mem){
      return NULL;
    }
};
