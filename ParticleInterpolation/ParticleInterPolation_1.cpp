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

#include "library/common/utilities.h"
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
    std::vector<double> &weights_out)
{


    // Smart shortcut
    auto order   = quadrature_order;
    auto mesh    = function->function_space()->mesh();
    auto dim     = mesh->topology().dim();
    auto element = function->function_space()->element();
    auto dofmap  = function->function_space()->dofmap();
    auto MPI_RANK = MPI::rank(MPI_COMM_WORLD);
    auto MPI_SIZE = MPI::size(MPI_COMM_WORLD);

    // Local variables, deleted at the end of the function
	std::vector<double> coordinates;
    std::vector<double> function_dofs;

    // assert for input valuables
    assert(values_out.size()==0);
    assert(points_out.size()==0);    
    assert(weights_out.size()==0);

    // Construct Gauss quadrature rule
    SimplexQuadrature gauss_quadrature(dim, order);

    std::vector<double> points_out_local;
    std::vector<double> weights_out_local;
    std::vector<double> function_dofs_local;
    std::vector<double> coordinates_local;

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        // Create ufc_cell associated with dolfin cell.
        ufc::cell ufc_cell;
        cell->get_cell_data(ufc_cell);

        // Compute quadrature rule for the cell.
        // push back gauss points and gauss weights.
        auto quadrature_rule = gauss_quadrature.compute_quadrature_rule(*cell);

        points_out_local.insert(points_out_local.end(),quadrature_rule.first.begin(),quadrature_rule.first.end());
        weights_out_local.insert(weights_out_local.end(),quadrature_rule.second.begin(),quadrature_rule.second.end());
        
        // push back function dofs on the cell.
        auto dofs = dofmap->cell_dofs(cell->index());
        std::vector<double> cell_function_dofs(dofs.size());
        function->vector()->get_local(cell_function_dofs.data(), dofs.size(), dofs.data());
        function_dofs_local.insert(function_dofs_local.end(),cell_function_dofs.begin(),cell_function_dofs.end());

        // push back cell coordinates.
        std::vector<double> cell_coordinates;
        cell->get_vertex_coordinates(cell_coordinates);
        coordinates_local.insert(coordinates_local.end(),cell_coordinates.begin(),cell_coordinates.end());
    }
    MPI::gather(MPI_COMM_WORLD, points_out_local, points_out, 0);
    MPI::gather(MPI_COMM_WORLD, weights_out_local, weights_out, 0);
    MPI::gather(MPI_COMM_WORLD, function_dofs_local, function_dofs, 0);
    MPI::gather(MPI_COMM_WORLD, coordinates_local, coordinates, 0);


    if (MPI_RANK == 0) { 
        size_t value_size = function->value_size();
        size_t num_cells  = mesh->num_entities_global(dim);
        size_t num_gauss  = weights_out.size()/num_cells;
        size_t num_dofs   = function_dofs.size()/num_cells;
        values_out.resize(points_out.size()/dim*value_size);
        // this parameter is true if gpu is used.
        PolynomialInterpolation pli(true);
        pli.evaluate_function(num_cells,num_gauss,value_size,num_dofs,coordinates.data(),
                            function_dofs.data(),points_out.data(),values_out.data());
        
    } else {
        points_out.clear();
        values_out.clear();
        weights_out.clear();
    }
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

    auto MPI_SIZE = MPI::size(MPI_COMM_WORLD);
    auto MPI_RANK = MPI::rank(MPI_COMM_WORLD);

    if(isSolid){
        assert(function->function_space()==displacement->function_space());
        std::vector<double> points;
        std::vector<double> weights;
        std::vector<double> values_function;
        std::vector<double> values_displace;
        evaluation_at_gauss_points(function, points, values_function, weights);
        weights.clear(); points.clear();
        evaluation_at_gauss_points(displacement, points, values_displace, weights);
        /// write the results back to the parameters.
        if (MPI_RANK == 0) {
            for(size_t i = 0; i < weights.size(); i++){
                coordinates.push_back(static_cast<float>(values_displace[3*i+0]));
                coordinates.push_back(static_cast<float>(values_displace[3*i+1]));
                coordinates.push_back(static_cast<float>(values_displace[3*i+2]));
                values_weights.push_back(static_cast<float>(values_function[3*i+0]));
                values_weights.push_back(static_cast<float>(values_function[3*i+1]));
                values_weights.push_back(static_cast<float>(values_function[3*i+2]));
                values_weights.push_back(static_cast<float>(weights[i]));
            }
        } else {
            assert(coordinates.size() == 0);
            assert(values_weights.size() == 0);
            assert(points.size() == 0);
            assert(weights.size() == 0);
            assert(values_displace.size() == 0);
            assert(values_function.size() == 0);
        }
    } 
    
    
    else {
        std::vector<double> points;
        std::vector<double> weights;
        std::vector<double> values;
        evaluation_at_gauss_points(function, points, values, weights);
        if (MPI_RANK==0){
            for(size_t i = 0; i<weights.size(); i++){
                coordinates.push_back(static_cast<float>(points[3*i+0]));
                coordinates.push_back(static_cast<float>(points[3*i+1]));
                coordinates.push_back(static_cast<float>(points[3*i+2]));
                values_weights.push_back(static_cast<float>(values[3*i+0]));
                values_weights.push_back(static_cast<float>(values[3*i+1]));
                values_weights.push_back(static_cast<float>(values[3*i+2]));
                values_weights.push_back(static_cast<float>(weights[i]));
            }
        } else {
            assert(coordinates.size() == 0);
            assert(values_weights.size() == 0);
            assert(points.size() == 0);
            assert(weights.size() == 0);
            assert(values.size() == 0);
        }
    }
    
    assert(values_weights.size() * 3 == coordinates.size() * 4);
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
    auto MPI_RANK = MPI::rank(MPI_COMM_WORLD);
    auto MPI_SIZE = MPI::size(MPI_COMM_WORLD);

    // generate points and values on mesh.    
    // find the minum point and maximun point.
    Point min, max;
    auto coord_f_local = f->function_space()->tabulate_dof_coordinates();
    auto coord_g_local = g->function_space()->tabulate_dof_coordinates();
    std::vector<double> coord_f;
    std::vector<double> coord_g;
    /// gather at processor zero.
    MPI::gather(MPI_COMM_WORLD, coord_f_local, coord_f, 0);   
    MPI::gather(MPI_COMM_WORLD, coord_g_local, coord_g, 0);
    if(MPI_RANK == 0) find_min_max_points(coord_f, coord_g, min, max);

    if(MPI_SIZE == 1) std::cout<<min<<max<<std::endl;
    else if (MPI_RANK == 0) std::cout<<min<<max<<std::endl;

     // calculate positions, values, weights(if isSolid is true,
    // f and d are on the solid and the pos_new will be evaluated on d)
    std::vector<float> pos_old;
    std::vector<float> val_old;
    get_gauss_rule(isSolid, f, d, pos_old, val_old);
    if(MPI_RANK != 0){
        assert(pos_old.size()==0);
        assert(val_old.size()==0);
    }

    // generate pos_new on mesh_new and allocate memory for val_new.
    // (if isSolid is true, f and d are solid and g is on fluid)
    std::vector<float> pos_new;
    std::vector<float> val_new;
    if (isSolid)
    {
        /// get the dof coordinates of fluid function.
        if (MPI_RANK == 0) {
            pos_new.resize(coord_g.size() / 3);
            for (size_t i = 0; i < coord_g.size() / 9; i++)
            {
                pos_new[3 * i] = static_cast<float>(coord_g[9 * i]);
                pos_new[3 * i + 1] = static_cast<float>(coord_g[9 * i + 1]);
                pos_new[3 * i + 2] = static_cast<float>(coord_g[9 * i + 2]);
            }
            val_new.resize(pos_new.size());
        } else {
            assert(pos_new.size() == 0);
            assert(val_new.size() == 0);
        }
    }
    else
    {
        /// get the dof coordinates of solid function.
        /// It should be the dofs of displacement function.
        std::vector<double> temp_pos_new;
        std::vector<float> pos_new_local;
        d->vector()->get_local(temp_pos_new);
        vectorTypeConvert(temp_pos_new,pos_new_local);
        MPI::gather(MPI_COMM_WORLD,pos_new_local, pos_new, 0);
        if(MPI_RANK == 0){
            val_new.resize(pos_new.size());
        } else {
            assert(val_new.size() == 0);
            assert(pos_new.size() == 0);
        }
    }
    
    if (MPI_RANK == 0){
        // use the particle system for delta interpolation.
        ParticleSystem particle_system(pos_old.size() / 3, radius, min[0], min[1], min[2], max[0] - min[0], max[1] - min[1], max[2] - min[2]);
        particle_system.inputData(pos_old.data(), val_old.data());
        particle_system.interpolate(pos_new.size() / 3, pos_new.data(), val_new.data());
    }

    /// distribute data on every processor.
    /// easier, faster, but cost more memory.
    /*
    val_new = my_mpi_gather(val_new);
    auto local_range = g->vector()->local_range();
    auto local_size  = g->vector()->local_size();
    auto offset = local_range.first;
    std::vector<double> val_local(local_size);
    for (size_t j = 0; j < local_size; j++) {
	    val_local[j] = val_new[offset+j];
	}
    g->vector()->set_local(val_local);
    g->vector()->apply("insert");
    */

    /// Try to use another way. This way cost less memory.
    
    /// collect the local range of every processor. 
    auto temp_range = g->vector()->local_range();
    std::vector<size_t> single_range{temp_range.first, temp_range.second};
    std::vector<size_t> global_range;
    MPI::gather(MPI_COMM_WORLD, single_range, global_range, 0);

    /// scatter values to every processor.
    std::vector<std::vector<double>> values_send;
    std::vector<double> values_receive;
    if (MPI_RANK == 0) {
        std::cout << "size : " << global_range.size() << std::endl;
        for(size_t i = 0; i<MPI_SIZE; i++){
            std::cout << "from " << global_range[2*i] << " to " << global_range[2*i+1] <<" in processor "<< i <<std::endl;
        }
        for(size_t i = 0; i < MPI_SIZE; i++){
            auto offset = global_range[2*i];
            auto local_size = global_range[2*i+1]-global_range[2*i];
            std::vector<double> temp_values(local_size);
            for (size_t j = 0; j < local_size; j++) {
		        temp_values[j] = val_new[offset+j];
	        }
            values_send.push_back(temp_values);
        }
    }
    std::cout<<"OL" <<std::endl;
    MPI::scatter(MPI_COMM_WORLD, values_send, values_receive, 0);
    g->vector()->set_local(values_receive);
    g->vector()->apply("insert");
}

PYBIND11_MODULE(ParticleInterpolationPybind, m)
{
    m.def("interpolate", &interpolate, "A function which adds two numbers");
}