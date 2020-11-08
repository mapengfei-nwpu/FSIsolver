

// This file contains C wrappers around the some of the CUDA API and the
// kernel functions so that they can be called from "particleSystem.cpp"


#include <cstdlib>
#include <cstdio>
#include <string.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>

#include <helper_functions.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_impl.cuh"

extern "C"
{

    void cudaInit(int argc, char **argv)
    {
        int devID;

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        devID = findCudaDevice(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("No CUDA Capable devices found, exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }

    void allocateArray(void **devPtr, size_t size)
    {
        checkCudaErrors(cudaMalloc(devPtr, size));
    }

    void freeArray(void *devPtr)
    {
        checkCudaErrors(cudaFree(devPtr));
    }

    void threadSync()
    {
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void copyArrayToDevice(void *device, void *host, int size)
    {
        checkCudaErrors(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
    }

    void copyArrayFromDevice(void* device, void* host, int size)
    {
        checkCudaErrors(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
    }

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    uint iDivUp(uint a, uint b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    void calcHash(uint  *gridParticleHash,
                  uint  *gridParticleIndex,
                  float *pos,
                  int    numParticles)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // execute the kernel
        calcHashD<<< numBlocks, numThreads >>>(gridParticleHash,
                                               gridParticleIndex,
                                               (float3 *) pos,
                                               numParticles);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");
    }

    void reorderDataAndFindCellStart(uint  *cellStart,
                                     uint  *cellEnd,
                                     float *sortedPos,
                                     float *sortedVel,
                                     uint  *gridParticleHash,
                                     uint  *gridParticleIndex,
                                     float *oldPos,
                                     float *oldVel,
                                     uint   numParticles,
                                     uint   numCells)
    {
        uint numThreads, numBlocks;
        computeGridSize(numParticles, 256, numBlocks, numThreads);

        // set all cells to empty
        checkCudaErrors(cudaMemset(cellStart, 0xffffffff, numCells*sizeof(uint)));

        uint smemSize = sizeof(uint)*(numThreads+1);
        reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(
            cellStart,
            cellEnd,
            (float3 *) sortedPos,
            (float4 *) sortedVel,
            gridParticleHash,
            gridParticleIndex,
            (float3 *) oldPos,
            (float4 *) oldVel,
            numParticles);
        getLastCudaError("Kernel execution failed: reorderDataAndFindCellStartD");

    }

    void collide(float *newVel,
                 float *newPos,
                 float *sortedVel,
                 float *sortedPos,
                 uint  *cellStart,
                 uint  *cellEnd,
                 uint   numParticles_new,
                 uint   numCells)
    {

        // thread per particle
        uint numThreads, numBlocks;
        computeGridSize(numParticles_new, 64, numBlocks, numThreads);

        // execute the kernel
        collideD<<< numBlocks, numThreads >>>((float3 *)newVel,
                                              (float3 *)newPos,
                                              (float4 *)sortedVel,
                                              (float3*)sortedPos,
                                              cellStart,
                                              cellEnd,
                                              numParticles_new);

        // check if kernel invocation generated an error
        getLastCudaError("Kernel execution failed");

    }


    void sortParticles(uint *dGridParticleHash, uint *dGridParticleIndex, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<uint>(dGridParticleHash),
                            thrust::device_ptr<uint>(dGridParticleHash + numParticles),
                            thrust::device_ptr<uint>(dGridParticleIndex));
    }

}   // extern "C"
