

#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#include "vector_types.h"
typedef unsigned int uint;

// simulation parameters
struct SimParams
{
    uint3 gridSize;
    float cellRadius;
    uint numCells;
    uint numParticles;
    float3 worldOrigin;
    float3 worldSize;
    float3 cellSize;
};

#endif
