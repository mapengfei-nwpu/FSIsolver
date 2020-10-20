// Includes
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <dolfin.h>
#include <dolfin/geometry/SimplexQuadrature.h>

#include "TetPoisson.h"
#include "particleSystem.h"

float radius;
size_t quadrature_order;

using namespace dolfin;

void find_min_max_points(
    const std::vector<double> &coord_1,
    const std::vector<double> &coord_2,
    Point &min,
    Point &max)
{
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::min();
    double max_y = std::numeric_limits<double>::min();
    double max_z = std::numeric_limits<double>::min();
    for (size_t i = 0; i < coord_1.size() / 3; i++)
    {
        min_x = coord_1[3 * i] < min_x ? coord_1[3 * i] : min_x;
        max_x = coord_1[3 * i] > max_x ? coord_1[3 * i] : max_x;
        min_y = coord_1[3 * i + 1] < min_y ? coord_1[3 * i + 1] : min_y;
        max_y = coord_1[3 * i + 1] > max_y ? coord_1[3 * i + 1] : max_y;
        min_z = coord_1[3 * i + 2] < min_z ? coord_1[3 * i + 2] : min_z;
        max_z = coord_1[3 * i + 2] > max_z ? coord_1[3 * i + 2] : max_z;
    }
    for (size_t i = 0; i < coord_2.size() / 3; i++)
    {
        min_x = coord_2[3 * i] < min_x ? coord_2[3 * i] : min_x;
        max_x = coord_2[3 * i] > max_x ? coord_2[3 * i] : max_x;
        min_y = coord_2[3 * i + 1] < min_y ? coord_2[3 * i + 1] : min_y;
        max_y = coord_2[3 * i + 1] > max_y ? coord_2[3 * i + 1] : max_y;
        min_z = coord_2[3 * i + 2] < min_z ? coord_2[3 * i + 2] : min_z;
        max_z = coord_2[3 * i + 2] > max_z ? coord_2[3 * i + 2] : max_z;
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
    auto order = quadrature_order;
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
    assert(values_weights.size() * 3 == coordinates.size() * 4);
}

// type conversion between std::vector<double> and std::vector<float>.
template <typename T1, typename T2>
void vectorTypeConvert(const std::vector<T1> &from, std::vector<T2> &to)
{
    to.resize(from.size());
    for (size_t i = 0; i < from.size(); i++)
    {
        to[i] = static_cast<T2>(from[i]);
    }
}

void interpolate(std::shared_ptr<const Function> f, std::shared_ptr<Function> g, float input_radius, int input_order)
{
    radius = input_radius;
    quadrature_order = input_order;

    // generate points and values on mesh.
    // TODO: using mesh coordinates is easier.
    const auto coord_f = f->function_space()->tabulate_dof_coordinates();
    const auto coord_g = g->function_space()->tabulate_dof_coordinates();

    // find the minum point and maximun point.
    Point min, max;
    find_min_max_points(coord_f, coord_g, min, max);

    // calculate positions, values, weights
    std::vector<float> pos_old;
    std::vector<float> val_old;
    get_gauss_rule(f, pos_old, val_old);

    // generate pos_new on mesh_new and allocate memory for val_new.
    std::vector<float> pos_new;
    std::vector<float> val_new;
    pos_new.resize(coord_g.size() / 3);
    for (size_t i = 0; i < coord_g.size() / 9; i++)
    {
        pos_new[3 * i] = static_cast<float>(coord_g[9 * i]);
        pos_new[3 * i + 1] = static_cast<float>(coord_g[9 * i + 1]);
        pos_new[3 * i + 2] = static_cast<float>(coord_g[9 * i + 2]);
    }
    val_new.resize(pos_new.size());

    // use the particle system for delta interpolation.
    ParticleSystem particle_system(pos_old.size() / 3, radius, min[0], min[1], min[2], max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    particle_system.inputData(pos_old.data(), val_old.data());
    particle_system.interpolate(pos_new.size() / 3, pos_new.data(), val_new.data());

    // convert the result from float to double and assign it to function.
    std::vector<double> val_new_double;
    vectorTypeConvert(val_new, val_new_double);
    g->vector()->set_local(val_new_double);
}

PYBIND11_MODULE(IB, m)
{
    m.def("interpolate", &interpolate, "A function which adds two numbers");
}