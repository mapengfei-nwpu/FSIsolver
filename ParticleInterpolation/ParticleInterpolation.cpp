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
#include "PolynomialInterpolation.h"

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
void evaluation_at_gauss_points(
    std::shared_ptr<const Function> function,
    std::vector<double> &points_out,
    std::vector<double> &values_out,
    std::vector<double> &weights_out,
    size_t order = 3)
{
    // Smart shortcut
    auto mesh    = function->function_space()->mesh();
    auto dim     = mesh->topology().dim();
    auto element = function->function_space()->element();
    auto dofmap  = function->function_space()->dofmap();

    // Local variables, deleted at the end of the function
	std::vector<double> coordinates;
    std::vector<double> function_dofs;

    // assert for input valuables
    assert(values_out.size()==0);
    assert(points_out.size()==0);    
    assert(weights_out.size()==0);

    // Construct Gauss quadrature rule
    SimplexQuadrature gauss_quadrature(dim, order);

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        // Create ufc_cell associated with dolfin cell.
        ufc::cell ufc_cell;
        cell->get_cell_data(ufc_cell);

        // Compute quadrature rule for the cell.
        // push back gauss points and gauss weights.
        auto quadrature_rule = gauss_quadrature.compute_quadrature_rule(*cell);

        points_out.insert(points_out.end(),quadrature_rule.first.begin(),quadrature_rule.first.end());
        weights_out.insert(weights_out.end(),quadrature_rule.second.begin(),quadrature_rule.second.end());
        
        // push back function dofs on the cell.
        auto dofs = dofmap->cell_dofs(cell->index());
        std::vector<double> cell_function_dofs(dofs.size());
        function->vector()->get(cell_function_dofs.data(), dofs.size(), dofs.data());
        function_dofs.insert(function_dofs.end(),cell_function_dofs.begin(),cell_function_dofs.end());

        // push back cell coordinates.
        std::vector<double> cell_coordinates;
        cell->get_vertex_coordinates(cell_coordinates);
        coordinates.insert(coordinates.end(),cell_coordinates.begin(),cell_coordinates.end());
    }

    size_t value_size = function->value_size();
    size_t num_cells  = mesh->num_cells();
    size_t num_gauss  = weights_out.size()/num_cells;
    size_t num_dofs   = function_dofs.size()/num_cells;
    values_out.resize(points_out.size()/dim*value_size);

    PolynomialInterpolation pli;
    pli.evaluate_function(num_cells,num_gauss,value_size,num_dofs,coordinates.data(),
                          function_dofs.data(),points_out.data(),values_out.data());
}

void get_gauss_rule(
    bool isSolid,
    std::shared_ptr<const Function> function,
    std::shared_ptr<const Function> displacement,
    std::vector<float> &coordinates,
    std::vector<float> &values_weights)
{
    assert(coordinates.size()==0);
    assert(values_weights.size()==0);
    if(isSolid){
        assert(function->function_space()==displacement->function_space());
        std::vector<double> points;
        std::vector<double> weights;
        std::vector<double> values_function;
        std::vector<double> values_displace;
        evaluation_at_gauss_points(function, points, values_function, weights);
        weights.resize(0);
        points.resize(0);
        evaluation_at_gauss_points(displacement, points, values_displace, weights);
        for(size_t i = 0; i<weights.size(); i++){
            coordinates.push_back(static_cast<float>(values_displace[3*i+0]));
            coordinates.push_back(static_cast<float>(values_displace[3*i+1]));
            coordinates.push_back(static_cast<float>(values_displace[3*i+2]));
            values_weights.push_back(static_cast<float>(values_function[3*i+0]));
            values_weights.push_back(static_cast<float>(values_function[3*i+1]));
            values_weights.push_back(static_cast<float>(values_function[3*i+2]));
            values_weights.push_back(static_cast<float>(weights[i]));
        }
    } else {
        std::vector<double> points;
        std::vector<double> weights;
        std::vector<double> values;
        evaluation_at_gauss_points(function, points, values, weights);
        for(size_t i = 0; i<weights.size(); i++){
            coordinates.push_back(static_cast<float>(points[3*i+0]));
            coordinates.push_back(static_cast<float>(points[3*i+1]));
            coordinates.push_back(static_cast<float>(points[3*i+2]));
            values_weights.push_back(static_cast<float>(values[3*i+0]));
            values_weights.push_back(static_cast<float>(values[3*i+1]));
            values_weights.push_back(static_cast<float>(values[3*i+2]));
            values_weights.push_back(static_cast<float>(weights[i]));
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

void interpolate(std::shared_ptr<const Function> f, // interpolation function
                 std::shared_ptr<const Function> d, // displacement function
                 std::shared_ptr<Function> g,       // unkown function
                 float input_radius,                // radius of cell
                 int input_order,                   // order of gauss quadrature
                 bool isSolid)                      // are f and d functions on solid?
{
    radius = input_radius;
    quadrature_order = input_order;

    // generate points and values on mesh.    
    // find the minum point and maximun point.
    // TODO: using mesh coordinates is easier.
    Point min, max;
    const auto coord_f = f->function_space()->tabulate_dof_coordinates();
    const auto coord_g = g->function_space()->tabulate_dof_coordinates();
    find_min_max_points(coord_f, coord_g, min, max);

    // calculate positions, values, weights(if isSolid is true,
    // f and d are on the solid and the pos_new will be evaluated on d)
    std::vector<float> pos_old;
    std::vector<float> val_old;
    get_gauss_rule(isSolid, f, d, pos_old, val_old);

    // generate pos_new on mesh_new and allocate memory for val_new.
    // (if isSolid is true, f and d are solid and g is on solid, pos_old
    // should be evaluated with instead)
    std::vector<float> pos_new;
    std::vector<float> val_new;
    if (isSolid)
    {
        pos_new.resize(coord_g.size() / 3);
        for (size_t i = 0; i < coord_g.size() / 9; i++)
        {
            pos_new[3 * i] = static_cast<float>(coord_g[9 * i]);
            pos_new[3 * i + 1] = static_cast<float>(coord_g[9 * i + 1]);
            pos_new[3 * i + 2] = static_cast<float>(coord_g[9 * i + 2]);
        }
        val_new.resize(pos_new.size());
    }
    else
    {
        std::vector<double> temp_pos_new;
        d->vector()->get_local(temp_pos_new);
        vectorTypeConvert(temp_pos_new,pos_new);
        val_new.resize(pos_new.size());
    }

    // use the particle system for delta interpolation.
    ParticleSystem particle_system(pos_old.size() / 3, radius, min[0], min[1], min[2], max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    particle_system.inputData(pos_old.data(), val_old.data());
    particle_system.interpolate(pos_new.size() / 3, pos_new.data(), val_new.data());

    // convert the result from float to double and assign it to function.
    std::vector<double> val_new_double;
    vectorTypeConvert(val_new, val_new_double);
    g->vector()->set_local(val_new_double);
}

PYBIND11_MODULE(ParticleInterpolationPybind, m)
{
    m.def("interpolate", &interpolate, "A function which adds two numbers");
}