
#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <algorithm>


ParticleSystem::ParticleSystem(uint numParticles, float cellRadius,
    float xOrigin, float yOrigin, float zOrigin,
    float xSize,   float ySize,   float zSize):
    m_bInitialized(false),
    m_numParticles(numParticles),
    m_dPos(nullptr),
    m_dVal(nullptr),
    m_dPos_new(nullptr),
    m_dVal_new(nullptr)
{
    // set the radius of a cell in the grid. 
    m_params.cellSize = make_float3(cellRadius, cellRadius, cellRadius);

    // set the simulation world as [-1��-1��-1]X[1��1��1].
    m_params.worldOrigin = make_float3(xOrigin, yOrigin, zOrigin);
    m_params.worldSize   = make_float3(xSize,   ySize,   zSize);

    // compute the grid size.
    auto num_cellx = static_cast<uint>(ceil(m_params.worldSize.x / cellRadius));
    auto num_celly = static_cast<uint>(ceil(m_params.worldSize.y / cellRadius));
    auto num_cellz = static_cast<uint>(ceil(m_params.worldSize.z / cellRadius));
    m_gridSize = make_uint3(num_cellx, num_celly, num_cellz);
    m_numGridCells = m_gridSize.x * m_gridSize.y * m_gridSize.z;

    // set simulation parameters.
    m_params.cellRadius = cellRadius;
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    m_params.numParticles = m_numParticles;

    _initialize(numParticles);
}


ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = 0;
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate GPU data

    allocateArray((void **)&m_dPos, sizeof(float) * 3 * m_numParticles);
    allocateArray((void **)&m_dVal, sizeof(float) * 4 * m_numParticles);

    allocateArray((void **)&m_dSortedPos, sizeof(float) * 3 * m_numParticles);
    allocateArray((void **)&m_dSortedVal, sizeof(float) * 4 * m_numParticles);

    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));

    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    freeArray(m_dPos);
    freeArray(m_dVal);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVal);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);
}

void ParticleSystem::inputData(float* pos, float* val) {

    copyArrayToDevice(m_dPos, pos, sizeof(float) * m_numParticles * 3);
    copyArrayToDevice(m_dVal, val, sizeof(float) * m_numParticles * 4);

    calcHash(m_dGridParticleHash, m_dGridParticleIndex, m_dPos, m_numParticles);

    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    reorderDataAndFindCellStart(m_dCellStart, m_dCellEnd, m_dSortedPos, m_dSortedVal,
        m_dGridParticleHash, m_dGridParticleIndex, m_dPos, m_dVal, m_numParticles, m_numGridCells);
}

void ParticleSystem::interpolate(uint numParticleNew, float* pos, float* val) {

    allocateArray((void**)&m_dPos_new, sizeof(float) * 3 * numParticleNew);
    allocateArray((void**)&m_dVal_new, sizeof(float) * 3 * numParticleNew);

    copyArrayToDevice(m_dPos_new, pos, sizeof(float) * 3 * numParticleNew);
    
    // collide
    collide(m_dVal_new, m_dPos_new, m_dSortedVal, m_dSortedPos,m_dCellStart,m_dCellEnd,
            numParticleNew,m_numGridCells);
    // finally write data
    copyArrayFromDevice(m_dVal_new, val, sizeof(float) * 3 * numParticleNew);

    freeArray(m_dPos_new);
    freeArray(m_dVal_new);
}
