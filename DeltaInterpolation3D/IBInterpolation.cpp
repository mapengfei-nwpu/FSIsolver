#include "IBInterpolation.h"
using namespace dolfin;

template <typename T>
std::vector<T> my_mpi_gather(std::vector<T> local)
{
	auto mpi_size = MPI::size(MPI_COMM_WORLD);

	/// collect local values on every process
	std::vector<std::vector<T>> mpi_collect(mpi_size);
	MPI::all_gather(MPI_COMM_WORLD, local, mpi_collect);
	std::vector<T> global;

	/// unwrap mpi_collect
	for (size_t i = 0; i < mpi_collect.size(); i++)
	{
		/// TODO : I failed to do the following step.
		/// global.insert(global.end(), mpi_collect[i].begin(), mpi_collect.end());
		for (size_t j = 0; j < mpi_collect[i].size(); j++)
		{
			global.push_back(mpi_collect[i][j]);
		}
	}
	return global;
}

void get_gauss_rule(
	std::shared_ptr<const Mesh> mesh,
	std::vector<double> &coordinates,
	std::vector<double> &weights)
{
	auto order = 6;
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
DeltaInterpolation::DeltaInterpolation(std::shared_ptr<IBMesh> fluid_mesh, 
									 std::shared_ptr<Mesh> solid_mesh,
					                 std::shared_ptr<Function> solid) 
:  fluid_mesh(fluid_mesh),
   solid_mesh(solid_mesh)
{
	side_lengths = fluid_mesh->cell_length();

	/// get gauss points of solid mesh.
	get_gauss_rule(solid_mesh, gauss_points_reference, weights);
	/// weights = my_mpi_gather(weights);
	gauss_points_current = gauss_points_reference;

	/// get dof points of solid mesh.
	auto dof_coordinates = solid->function_space()->tabulate_dof_coordinates();

	/// guarantee all entries of dofmap are in local processor.
	assert(dof_coordinates.size() % 9 == 0);

	for (size_t i = 0; i < dof_coordinates.size(); i += 9)
	{
		dof_points_current.push_back(dof_coordinates[i]);
		dof_points_current.push_back(dof_coordinates[i + 1]);
		dof_points_current.push_back(dof_coordinates[i + 2]);
	}

	/// get the map from gauss points to cell.
	for (size_t i = 0; i < gauss_points_reference.size(); i += 3)
	{
		Point point(gauss_points_reference[i],
					gauss_points_reference[i + 1],
					gauss_points_reference[i + 2]);
		unsigned int id = solid_mesh->bounding_box_tree()->compute_first_entity_collision(point);
		gauss_points_to_cell.push_back(id);
	}
}
void DeltaInterpolation::fluid_to_solid(Function &fluid, Function &solid)
{
	/// Smart shortcut
	auto mesh = fluid.function_space()->mesh();
	auto local_size = solid.vector()->local_size();
	auto mpi_rank = MPI::rank(MPI_COMM_WORLD);
	auto mpi_size = MPI::size(MPI_COMM_WORLD);

	/// Initial local polynomial object. If the function is wanted to be
	/// evaluated on a point in a cell, local_poly will construct a polynomial
	/// function on this cell and store its coefficients.
	LocalPolynomial local_poly;
	local_poly.function = fluid;

	/// calculate global dof coordinates and dofs of solid.
	std::vector<double> local_values(local_size);

	/// mpi steps
	std::vector<double> points_send = dof_points_current;
	std::vector<double> values_send = local_values;
	std::vector<double> points_receive = dof_points_current;
	std::vector<double> values_receive = local_values;

	/// loop all dof points
	for (std::size_t i = 0; i < mpi_size; i++)
	{
		MPI::send_recv(MPI_COMM_WORLD, points_send, (mpi_rank + 1) % mpi_size, points_receive, (mpi_rank + mpi_size - 1) % mpi_size);
		MPI::send_recv(MPI_COMM_WORLD, values_send, (mpi_rank + 1) % mpi_size, values_receive, (mpi_rank + mpi_size - 1) % mpi_size);
		for (std::size_t j = 0; j < points_receive.size() / 3; ++j)
		{
			Point point(
				points_receive[j * 3],
				points_receive[j * 3 + 1],
				points_receive[j * 3 + 2]);
			auto hash_index = fluid_mesh->hash(point);
			auto mpi_rank_and_local_index = fluid_mesh->global_map[hash_index];
			if (mpi_rank_and_local_index[0] == mpi_rank)
			{
				/// Find the dolfin cell where point reside
				auto local_index = mpi_rank_and_local_index[1];
				auto values = local_poly.eval(point, local_index);
				
				values_receive[j * 3] = values[0];
				values_receive[j * 3 + 1] = values[1];
				values_receive[j * 3 + 2] = values[2];
			}
		}
		points_send = points_receive;
		values_send = values_receive;
	}

	local_values = values_receive;
	solid.vector()->set_local(local_values);
	solid.vector()->apply("insert");
}

void DeltaInterpolation::solid_to_fluid(Function &fluid, Function &solid)
{

	/// Set all entries to zero. 
	fluid.vector()->zero();

	/// Set MPI variables.
	auto mpi_size = MPI::size(MPI_COMM_WORLD);
	auto mpi_rank = MPI::rank(MPI_COMM_WORLD);

	/// These are weights, gauss points, weights on local processor.
	std::vector<double> points_send;
	std::vector<double> values_send;
	std::vector<double> weights_send;

	std::vector<double> points_receive;
	std::vector<double> values_receive;
	std::vector<double> weights_receive;

	/// calculate global dof coordinates and dofs of solid.
	std::vector<double> solid_values;
	for (size_t i = 0; i < gauss_points_reference.size() / 3; ++i)
	{
		Array<double> x(3, &(gauss_points_reference[i * 3]));
		Array<double> v(3);

		// Create cell that contains point
		const Cell cell(*solid_mesh, gauss_points_to_cell[i]);
		ufc::cell ufc_cell;
		cell.get_cell_data(ufc_cell);

		// Call evaluate function
		solid.eval(v, x, cell, ufc_cell);
		solid_values.push_back(v[0]);
		solid_values.push_back(v[1]);
		solid_values.push_back(v[2]);
	}

	values_send = solid_values;
	weights_send = weights;
	points_send = gauss_points_current;

	values_receive = solid_values;
	weights_receive = weights;
	points_receive = gauss_points_current;

	for (size_t i = 0; i < mpi_size; i++)
	{
		/// if(mpi_rank == 0)
		/// std::cout << ". mpi_rank: " << mpi_rank << "\n"
		/// 		  << "points_send.size(): " << points_send.size() << "\n"
		/// 		  << "weights_send.size(): " << weights_send.size() << "\n"
		/// 		  << "values_send.size(): " << values_send.size() << "\n"
		/// 		  << "gauss_points_reference: " << gauss_points_reference.size() << "\n"
		/// 		  << "gauss_points_current: " << gauss_points_current.size() << "\n"
		/// 		  << ". step: "     << i
		/// 		  << "\n from " << (mpi_rank+mpi_size-1)%mpi_size
		/// 		  << " to " << (mpi_rank+1)%mpi_size
		/// 		  << std::endl;
		/// send current vector to next vector.
		/// at the start : 0 1 2 3 4
		/// i = 0          4 0 1 2 3
		/// i = 1   	   3 4 0 1 2
		/// i = 2          2 3 4 0 1
		/// i = 3          1 2 3 4 0
		///		________________________
		///	   |                        |
		/// receive receive receive     |
		///    |  \   |   /  |          |
		///    |   \  |  /   |          |
		///  send    send   send _______|
		MPI::send_recv(MPI_COMM_WORLD, points_send, (mpi_rank + 1) % mpi_size, points_receive, (mpi_rank + mpi_size - 1) % mpi_size);
		MPI::send_recv(MPI_COMM_WORLD, weights_send, (mpi_rank + 1) % mpi_size, weights_receive, (mpi_rank + mpi_size - 1) % mpi_size);
		MPI::send_recv(MPI_COMM_WORLD, values_send, (mpi_rank + 1) % mpi_size, values_receive, (mpi_rank + mpi_size - 1) % mpi_size);
		solid_to_fluid_raw(fluid, values_receive, points_receive, weights_receive);

		values_send = values_receive;
		weights_send = weights_receive;
		points_send = points_receive;
	}

	/// copy boundary values from neighbour processors
	fluid.vector()->apply("insert");
}

void DeltaInterpolation::solid_to_fluid_raw(
	Function &fluid,
	std::vector<double> &solid_values,
	std::vector<double> &solid_coordinates,
	std::vector<double> &weights)
{
	/// smart shortcut
	auto mesh = fluid.function_space()->mesh();		// pointer to a mesh
	auto dofmap = fluid.function_space()->dofmap(); // pointer to a dofmap
	auto value_size = fluid.value_size();

	/// get the element of function space
	auto element = fluid.function_space()->element();

	/// Define vector size.
	auto offset_start = fluid.vector()->local_range().first;
	auto offset_end = fluid.vector()->local_range().second;
	auto local_size = offset_end - offset_start;

	/// Get local to global dofmap
	std::vector<size_t> local_to_global;
	dofmap->tabulate_local_to_global_dofs(local_to_global);

	/// if(MPI::rank(MPI_COMM_WORLD) == 3){
	/// 		for (size_t i = 0; i < local_size; i++)
	/// 		{
	/// 			if (local_to_global[i] != i + offset_start)
	/// 			std::cout << "local dof index : " << i + offset_start << "\n"
	/// 					  << "global dof index : " << local_to_global[i] << "\n"
	/// 					  << std::endl;
	/// 		}
	///
	/// 	}

	/// initial local fluid values.
	std::vector<double> local_values(local_size);

	/// iterate every gauss point of solid mesh.
	for (size_t i = 0; i < solid_values.size() / value_size; i++)
	{

		/// get indices of adjacent cells on fluid mesh.
		Point solid_point(solid_coordinates[3 * i], solid_coordinates[3 * i + 1], solid_coordinates[3 * i + 2]);
		auto adjacents = fluid_mesh->get_adjacents(solid_point, bandwidth);

		/// PERFORMANCE :
		/// it is not very efficient to use map instead of vector.
		/// Because different cells might share same dofs, we need to remove repeated dofs.
		/// So, it will be more complicated to use vector.

		std::map<size_t, double> indices_to_delta;

		/// iterate adjacent cells and collect element nodes in these cells.
		/// it has nothing to do with cell type.
		for (size_t j = 0; j < adjacents.size(); j++)
		{
			/// step 1 : get coordinates of cell dofs
			Cell cell(*mesh, adjacents[j]);
			std::vector<double> coordinate_dofs;
			cell.get_coordinate_dofs(coordinate_dofs);
			boost::multi_array<double, 2> coordinates;
			element->tabulate_dof_coordinates(coordinates, coordinate_dofs, cell);

			/// step 2 : get the dof map
			auto cell_dofmap = dofmap->cell_dofs(cell.index());

			/// step 3 : iterate node coordinates of the cell.
			for (size_t k = 0; k < cell_dofmap.size() / value_size; k++)
			{
				/// if(cell_dofmap[k+54] - cell_dofmap[k] != 2 ){
				/// 	std::cout
				/// 			  << cell_dofmap[k] <<" , "<< cell_dofmap[k+27] <<" , "<< cell_dofmap[k+54] << "\n"
				///				  << cell_dofmap[k]+offset_start - local_to_global[cell_dofmap[k]] << "\n"
				///				  << cell_dofmap[k+27]+offset_start - local_to_global[cell_dofmap[k+27]] << "\n"
				///				  << cell_dofmap[k+54]+offset_start - local_to_global[cell_dofmap[k+54]] << "\n"
				///				  << std::endl;
				///	}

				double param = delta(solid_point.coordinates(), &(coordinates.data()[3 * k]));
				if (param > 0.0)
				{
					indices_to_delta[cell_dofmap[k]] = param;
				}
			}
		}

		for (auto it = indices_to_delta.begin(); it != indices_to_delta.end(); it++)
		{
			for (size_t l = 0; l < value_size; l++)
			{
				auto local_index = local_to_global[it->first + l] - offset_start;
				if (local_index < local_size)
				{
					local_values[local_index] += solid_values[i * value_size + l] * it->second * weights[i];
				}
			}
		}
	}
	Array<double> local_values_array(local_values.size(), local_values.data());
	fluid.vector()->add_local(local_values_array);
}

void DeltaInterpolation::evaluate_current_points(std::shared_ptr<const Function> disp)
{
	/// TODO :
	/// The size of reference and current points should be the same.
	update(disp, gauss_points_to_cell, gauss_points_reference, gauss_points_current);
	disp->vector()->get_local(dof_points_current);
}

void DeltaInterpolation::update(
	std::shared_ptr<const Function> disp,
	const std::vector<size_t> &points_to_cell,
	const std::vector<double> &reference_points,
	std::vector<double> &current_points)
{
	/// re-create current_points
	/// current_points.resize(reference_points.size());
	for (size_t i = 0; i < reference_points.size() / 3; ++i)
	{
		Array<double> x(3);
		Array<double> v(3);
		x[0] = reference_points[i * 3];
		x[1] = reference_points[i * 3 + 1];
		x[2] = reference_points[i * 3 + 2];

		// Create cell that contains point
		const Cell cell(*solid_mesh, points_to_cell[i]);
		ufc::cell ufc_cell;
		cell.get_cell_data(ufc_cell);

		// Call evaluate function
		disp->eval(v, x, cell, ufc_cell);
		current_points[i * 3] = v[0];
		current_points[i * 3 + 1] = v[1];
		current_points[i * 3 + 2] = v[2];
	}
	/// re-assign current_points
	/// current_points = my_mpi_gather(current_points);
}