

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include "vector_functions.h"
#include "particles_kernel.cuh"
// Particle system class
class ParticleSystem
{
    public:
        // These parameters are enough to construct a background grid.
        ParticleSystem(
            uint numParticles, float cellRadius,
            float xOrigin, float yOrigin, float zOrigin,
            float xSize,   float ySize,   float zSize);
        
        // default background grid is [-1,-1,-1]x[1,1,1]
        ParticleSystem(uint numParticles, float cellRadius):
            ParticleSystem(numParticles, cellRadius, -1.0f, -1.0f, -1.0f, 2.0f, 2.0f, 2.0f)    
        {
            //
        }
        
        // default background with cell raius 0.2. 
        ParticleSystem(uint numParticles):
            ParticleSystem(numParticles, 0.2)
        {
            //
        }

        void inputData(float* pos, float* val);
        void interpolate(uint numParticleNew, float* pos, float* val);
        ~ParticleSystem();

    protected: // methods
        ParticleSystem() {}

        void _initialize(int numParticles);
        void _finalize();

    protected: // data
        bool m_bInitialized;
        uint m_numParticles;

        // GPU data
        float *m_dPos;
        float *m_dVal;
        float *m_dPos_new;
        float *m_dVal_new;

        float *m_dSortedPos;
        float *m_dSortedVal;

        // grid data for sorting method
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell

        // params
        SimParams m_params;
        uint3 m_gridSize;
        uint m_numGridCells;

};

#endif // __PARTICLESYSTEM_H__
