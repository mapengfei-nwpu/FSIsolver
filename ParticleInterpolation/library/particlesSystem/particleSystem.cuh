

extern "C"
{
    void cudaInit(int argc, char **argv);

    void allocateArray(void **devPtr, size_t size);
    void freeArray(void *devPtr);

    void threadSync();

    void copyArrayToDevice(void* device, void* host, int size);
    void copyArrayFromDevice(void* device, void* host, int size);
    void setParameters(SimParams *hostParams);


    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles);

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells);

    void collide(float* newVel,
        float* newPos,
        float* sortedVel,
        float* sortedPos,
        uint* cellStart,
        uint* cellEnd,
        uint   numParticles_new,
        uint   numCells);

    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles);

}
