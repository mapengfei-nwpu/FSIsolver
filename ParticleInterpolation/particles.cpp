// Includes
#define NDEBUG
#include <assert.h>
#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>

#include <dolfin.h>
#include <dolfin/geometry/SimplexQuadrature.h>

#include "particleSystem.h"

const uint num_new = 12582912;
const float radius = 0.3;

using namespace dolfin;

void data_generate(std::vector<float> &pos_new)
{
    // lambda function of random.
    srand(static_cast<uint>(time(0)));
    auto rrr = [] { return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX); };

    // random positions.
    for (size_t i = 0; i < pos_new.size(); i++)
    {
        pos_new[i] = rrr() * 0.3;
    }
}

void find_min_max_points(const std::vector<double> &coord, Point &min, Point &max)
{
    double min_x = 10000;
    double min_y = 10000;
    double min_z = 10000;
    double max_x = -10000;
    double max_y = -10000;
    double max_z = -10000;
    for (size_t i = 0; i < coord.size() / 3; i++)
    {
        min_x = coord[3 * i] < min_x ? coord[3 * i] : min_x;
        max_x = coord[3 * i] > max_x ? coord[3 * i] : max_x;
        min_y = coord[3 * i + 1] < min_y ? coord[3 * i + 1] : min_y;
        max_y = coord[3 * i + 1] > max_y ? coord[3 * i + 1] : max_y;
        min_z = coord[3 * i + 2] < min_z ? coord[3 * i + 2] : min_z;
        max_z = coord[3 * i + 2] > max_z ? coord[3 * i + 2] : max_z;
    }
    min[0] = min_x;
    min[1] = min_y;
    min[2] = min_z;
    max[0] = max_x;
    max[1] = max_y;
    max[2] = max_z;
}

void get_gauss_rule(
	std::shared_ptr<const Mesh> mesh,
	std::vector<double> &coordinates,
	std::vector<double> &weights)
{
	auto order = 1;
	auto dim = mesh->topology().dim();

	// Construct Gauss quadrature rule
	SimplexQuadrature gq(dim, order);

	for (CellIterator cell(*mesh); !cell.end(); ++cell)
	{
		// Create ufc_cell associated with dolfin cell.
		ufc::cell ufc_cell;
		cell->get_cell_data(ufc_cell);

		// Compute quadrature rule for the cell.
		auto qr = gq.compute_quadrature_rule(*cell);
		dolfin_assert(qr.second.size() == qr.first.size() / 3);
		/// std::cout << qr.first.size() << std::endl;
		for (size_t i = 0; i < qr.second.size(); i++)
		{
			/// push back what we get.
			weights.push_back(qr.second[i]);
			for (size_t d = 0; d < dim; d++)
			{
				coordinates.push_back(qr.first[3 * i + d]);
			}
		}
	}
}

void fun(std::shared_ptr<const Mesh> mesh, std::vector<float> &pos_old, std::vector<float> &val_old){
	std::vector<double> coordinates;
	std::vector<double> weights;
    get_gauss_rule(mesh, coordinates, weights);
    pos_old.resize(3*weights.size());
    val_old.resize(4*weights.size());
    for (size_t i = 0; i < weights.size(); i++)
    {
        float x = static_cast<float>(coordinates[3*i]);
        float y = static_cast<float>(coordinates[3*i+1]);
        float z = static_cast<float>(coordinates[3*i+2]);

        pos_old[3*i]   = x;
        pos_old[3*i+1] = y;
        pos_old[3*i+2] = z;

        val_old[4*i]   = 2.0;
        val_old[4*i+1] = 3.0;
        val_old[4*i+2] = 4*y;
        val_old[4*i+3] = weights[i];
    }
}

int main()
{
    auto mesh = std::make_shared<Mesh>(BoxMesh(Point(-2, -3, -2), Point(2, 2, 2), 128, 128, 128));
    const auto coord = mesh->coordinates();
    Point min, max;
    find_min_max_points(coord, min, max);
    std::cout << "min" << min << std::endl;
    std::cout << "max" << max << std::endl;
    std::cout << "size" << coord.size() / 3 << std::endl;

    // calculate values and weights
    std::vector<float> pos_old;
    std::vector<float> val_old;
    std::vector<float> pos_new;
    std::vector<float> val_new;
    
    fun(mesh, pos_old, val_old);

    for (size_t i = 0; i < val_old.size()/4; i++)
    {
        
        /// std::cout<<"0: "<<val_old[4*i]<<",1: "<<val_old[4*i+1]<<",2: "<<val_old[4*i+2]<<",3: "<<val_old[4*i+3]<<std::endl;
    }
    std::cout<<"size of weights:"<<val_old.size()<<std::endl;
    
    pos_new.resize(3*num_new);
    val_new.resize(3*num_new);
    data_generate(pos_new);

    assert(pos_old.size() * 4 == val_old.size() * 3);
    assert(pos_new.size() == val_new.size());

    auto start = std::chrono::system_clock::now();

    /// use the particle system for delta interpolation.
    ParticleSystem particle_system(pos_old.size() / 3, radius, min[0],min[1],min[2],max[0]-min[0],max[1]-min[1],max[2]-min[2]);
    particle_system.inputData(pos_old.data(), val_old.data());

    /// generate random particles
    particle_system.interpolate(pos_new.size() / 3, pos_new.data(), val_new.data());

    // do something...
    auto end   = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout <<  "花费了" 
              << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den   
              << "秒" << std::endl;

    /// print the results.
    for (size_t i = 0; i < pos_new.size() / 3; i++)
    {
        //printf("pos: %f, %f, %f\n", pos_new[3 * i], pos_new[3 * i + 1], pos_new[3 * i + 2]);
        //printf("val: %f, %f, %f\n", val_new[3 * i], val_new[3 * i + 1], val_new[3 * i + 2]);
    }
    std::cout<<"particles size:"<<val_old.size()<<std::endl;

    // here, val_new have been rewritten.
    return 0;
}
