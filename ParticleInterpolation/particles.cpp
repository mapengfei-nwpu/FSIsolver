// Includes
#define NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <time.h>

#include "particleSystem.h"
/// 一个奇怪的现象，如果 edge_divide 是 5 的倍数，那么最后的插值结果会比较精确。
/// 这和粒子的坐标的是否准确有关。
const uint edge_divide = 19;
const uint num_old = edge_divide*edge_divide*edge_divide;
const uint num_new = 600;
const float radius = 0.2;

void data_generate(std::vector<float> &pos_old,
                   std::vector<float> &pos_new,
                   std::vector<float> &val_old) {

    for (size_t i = 0; i < edge_divide; i++)
    {
        for (size_t j = 0; j < edge_divide; j++)
        {
            for (size_t k = 0; k < edge_divide; k++)
            {
                auto n = i * edge_divide * edge_divide + j * edge_divide + k;
                pos_old[n * 3    ] = i / static_cast<float>(edge_divide+1) + 0.5 / static_cast<float>(edge_divide+1)-0.5;
                pos_old[n * 3 + 1] = j / static_cast<float>(edge_divide+1) + 0.5 / static_cast<float>(edge_divide+1)-0.5;
                pos_old[n * 3 + 2] = k / static_cast<float>(edge_divide+1) + 0.5 / static_cast<float>(edge_divide+1)-0.5;
            }
        }
    }

    // grid positions and weights.
    srand(static_cast<uint>(time(0)));
    auto rrr = [] {return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); };
    for (size_t i = 0; i < val_old.size()/4; i++)
    {
        val_old[i * 4    ] = 4.0;
        val_old[i * 4 + 1] = 3.0;
        val_old[i * 4 + 2] = 2.0;
        val_old[i * 4 + 3] = 8.0 / static_cast<float>((edge_divide + 1)*(edge_divide + 1)*(edge_divide + 1)) / (radius * radius * radius);
    }

    // random positions.
    for (size_t i = 0; i < pos_new.size(); i++)
    {
        pos_new[i] = rrr()*0.3;
    }
}


int main() {
    std::vector<float> pos_old(3*num_old);
    std::vector<float> val_old(4*num_old);
    std::vector<float> pos_new(3*num_new);
    std::vector<float> val_new(3*num_new);

    data_generate(pos_old, pos_new, val_old);
    
    assert(pos_old.size() * 4 == val_old.size() * 3);
    assert(pos_new.size() == val_new.size());

    /// use the particle system for delta interpolation.
    ParticleSystem particle_system(pos_old.size() / 3, radius);
    particle_system.inputData(pos_old.data(), val_old.data());
    particle_system.interpolate(pos_new.size() / 3, pos_new.data(), val_new.data());
    
    /// print the results.
    for (size_t i = 0; i < pos_new.size() / 3; i++)
    {
        printf("pos: %f, %f, %f\n", pos_new[3 * i], pos_new[3 * i + 1], pos_new[3 * i + 2]);
        printf("val: %f, %f, %f\n", val_new[3 * i], val_new[3 * i + 1], val_new[3 * i + 2]);
    }
    
    // here, val_new have been rewritten.
    return 0;
}
