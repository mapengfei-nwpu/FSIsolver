// Includes
#define NDEBUG
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>

#include <dolfin.h>
#include <dolfin/geometry/SimplexQuadrature.h>

#include "TetPoisson.h"
#include "particleSystem.h"

const uint num_new = 100;
const float radius = 0.3;

using namespace dolfin;

class Source : public Expression
{
public:
    Source() : Expression(3) {}

    void eval(Array<double> &values, const Array<double> &x) const
    {
        values[0] = 3;
        values[1] = 3;
        values[2] = 3.0;
    }
};

std::shared_ptr<Function> generate_function(std::shared_ptr<const Mesh> mesh){
    // Create function space
    auto V = std::make_shared<TetPoisson::FunctionSpace>(mesh);
    auto f = std::make_shared<Function>(V);
    auto s = std::make_shared<Source>();
    f->interpolate(*s);
    return f;
}

void find_min_max_points(const std::vector<double> &coord, Point &min, Point &max)
{
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::min();
    double max_y = std::numeric_limits<double>::min();
    double max_z = std::numeric_limits<double>::min();
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
    std::shared_ptr<const Function> function,
    std::vector<float> &coordinates,
    std::vector<float> &values_weights)
{
    auto order = 1;
    auto mesh = function->function_space()->mesh();
    auto dim = mesh->topology().dim();

    // Construct Gauss quadrature rule
    SimplexQuadrature gauss_quadrature(dim, order);

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        // Create ufc_cell associated with dolfin cell.
        ufc::cell ufc_cell;
        cell->get_cell_data(ufc_cell);

        // Compute quadrature rule for the cell.
        auto quadrature_rule = gauss_quadrature.compute_quadrature_rule(*cell);
        assert(quadrature_rule.second.size() == quadrature_rule.first.size() / 3);

        // compute function values at quafrature points.
        for (size_t i = 0; i < quadrature_rule.second.size(); i++)
        {
            // shortcut of gauss point and weight.
            auto point = &(quadrature_rule.first[3 * i]);
            auto weight = quadrature_rule.second[i];

		    // Call evaluate function
            Array<double> x(3, point);
		    Array<double> v(3);
		    function->eval(v, x, *cell, ufc_cell);

            // push back gauss point.
            coordinates.push_back(static_cast<float>(point[0]));
            coordinates.push_back(static_cast<float>(point[1]));
            coordinates.push_back(static_cast<float>(point[2]));

            // Push back values and weights
		    values_weights.push_back(static_cast<float>(v[0]));
		    values_weights.push_back(static_cast<float>(v[1]));
		    values_weights.push_back(static_cast<float>(v[2]));
            values_weights.push_back(static_cast<float>(weight));
        }
    }
    assert(values_weights.size()*3==coordinates.size()*4);
}

// type conversion between std::vector<double> and std::vector<float>.
template<typename T1, typename T2>
void vectorTypeConvert(const std::vector<T1> &from, std::vector<T2> &to){
    to.resize(from.size());
    for (size_t i = 0; i < from.size(); i++)
    {
        to[i] = static_cast<T2>(from[i]);
    }
}

int main()
{
    auto mesh = std::make_shared<Mesh>(BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), 30, 30, 30));
    const auto coord = mesh->coordinates();
    Point min, max;
    find_min_max_points(coord, min, max);

    // calculate values and weights
    std::vector<float> pos_old;
    std::vector<float> val_old;

    // generate points and values on mesh.
    auto f = generate_function(mesh);
    get_gauss_rule(f,pos_old,val_old);

    // generate random points
    auto mesh_new = std::make_shared<Mesh>(BoxMesh(Point(-2, -2, -2), Point(2, 2, 2), 30, 30, 30));
    auto g = generate_function(mesh_new);
    auto dof_coordinates = g->function_space()->tabulate_dof_coordinates();

    // generate points and values on mesh_new.
    std::vector<float> pos_new;
    std::vector<float> val_new;
    pos_new.resize(dof_coordinates.size()/3);
    for (size_t i = 0; i < dof_coordinates.size()/9; i++)
    {
        pos_new[3*i]   = static_cast<float>(dof_coordinates[9*i]);
        pos_new[3*i+1] = static_cast<float>(dof_coordinates[9*i+1]);
        pos_new[3*i+2] = static_cast<float>(dof_coordinates[9*i+2]);
    }
    val_new.resize(pos_new.size());


    // time the interpolation
    auto start = std::chrono::system_clock::now();

    /// use the particle system for delta interpolation.
    ParticleSystem particle_system(pos_old.size() / 3, radius, min[0], min[1], min[2], max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    particle_system.inputData(pos_old.data(), val_old.data());

    /// generate random particles
    particle_system.interpolate(pos_new.size() / 3, pos_new.data(), val_new.data());

    // do something...
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "花费了"
              << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den
              << "秒" << std::endl;

    /// print the results.
    for (size_t i = 0; i < pos_new.size() / 3; i++)
    {
       // printf("pos: %f, %f, %f\n", pos_new[3 * i], pos_new[3 * i + 1], pos_new[3 * i + 2]);
       // printf("val: %f, %f, %f\n", val_new[3 * i], val_new[3 * i + 1], val_new[3 * i + 2]);
    }

    std::vector<double> val_new_double;
    vectorTypeConvert(val_new,val_new_double);

    
    std::cout<<"g->vector()->size(): "<<g->vector()->size()<<std::endl;
    std::cout<<"val_new.size(): "<<val_new.size()<<std::endl;
    std::cout<<"val_new_double.size(): "<<val_new_double.size()<<std::endl;
    g->vector()->set_local(val_new_double);


    File file("g.pvd");
    file << *g;


    
    
    return 0;
}
