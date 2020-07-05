#include <dolfin.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include <numeric>
#include "IBMesh.h"
using namespace dolfin;

template <typename T>
std::vector<T> my_mpi_gather(std::vector<T> local)
{

	auto mpi_size = dolfin::MPI::size(MPI_COMM_WORLD);

	/// collect local coordinate_dofs on every process
	std::vector<std::vector<T>> mpi_collect(mpi_size);
	dolfin::MPI::all_gather(MPI_COMM_WORLD, local, mpi_collect);
	std::vector<T> global;

	/// unwrap mpi_dof_coordinates.
	for (size_t i = 0; i < mpi_collect.size(); i++)
	{
		for (size_t j = 0; j < mpi_collect[i].size(); j++)
		{
			global.push_back(mpi_collect[i][j]);
		}
	}
	return global;
}

void get_gauss_rule(
	const std::shared_ptr<Mesh> mesh,
	std::vector<double> &points,
	std::vector<double> &weights)
{
	// Construct Gauss quadrature rules
	SimplexQuadrature gq(2, 6);

	/// iterate cell of local mesh.
	for (CellIterator cell(*mesh); !cell.end(); ++cell)
	{
		// Compute quadrature rule for the cell.
		auto qr = gq.compute_quadrature_rule(*cell);
		dolfin_assert(qr.second.size() == qr.first.size() / 2);
		for (size_t i = 0; i < qr.second.size(); i++)
		{
			/// push back what we get.
			points.push_back(qr.first[2*i]);
			points.push_back(qr.first[2*i+1]);
			weights.push_back(qr.second[i]);
		}
	}
	weights = my_mpi_gather(weights);
	points = my_mpi_gather(points);
}

class IBInterpolation
{
public:
	/// information about mesh structure.
	std::shared_ptr<IBMesh> fluid_mesh;
	std::shared_ptr<Mesh> solid_mesh;

	/// Define solid coordinates of gauss points and their weights.
	std::vector<double> current_gauss_points;
	std::vector<double> reference_gauss_points;
	std::vector<double> weights;

	/// Define solid coordinates of dofs.
	std::vector<double> current_dof_points;
	std::vector<double> reference_dof_points;

	/// Define the length of sides of fluid cell.
	std::vector<double> side_lengths;

	/// construct function.
	IBInterpolation(std::shared_ptr<IBMesh> fluid_mesh, 
					std::shared_ptr<Mesh> solid_mesh,
					std::shared_ptr<Function> solid) : 
		fluid_mesh(fluid_mesh),
		solid_mesh(solid_mesh)
	{
		side_lengths = fluid_mesh->side_length();
		fluid_mesh->set_bandwidth(1);
		
		///
		get_gauss_rule(solid_mesh, reference_gauss_points, weights);
		current_gauss_points.resize(reference_gauss_points.size());

		/// 
		auto dof_coordinates = solid->function_space()->tabulate_dof_coordinates();
		for (size_t i = 0; i < dof_coordinates.size(); i += 4)
		{
			reference_dof_points.push_back(dof_coordinates[i]);
			reference_dof_points.push_back(dof_coordinates[i + 1]);
		}
		current_dof_points.resize(reference_dof_points.size());
	}
	

	void evaluate_current_points(const std::shared_ptr<Function> disp)
	{
		/// TODO : the size of reference and current points
		///        should be the same.
		for(size_t i=0; i < current_gauss_points.size()/2; ++i)
		{
			Array<double> x(2);
			Array<double> v(2);
			x[0] = reference_gauss_points[i*2];
			x[1] = reference_gauss_points[i*2+1];
			disp->eval(v, x);
			current_gauss_points[i*2] = v[0];
			current_gauss_points[i*2+1] = v[1];
		}
		for(size_t i=0; i < reference_dof_points.size()/2; ++i)
		{
			Array<double> x(2);
			Array<double> v(2);
			x[0] = reference_dof_points[i*2];
			x[1] = reference_dof_points[i*2+1];
			disp->eval(v, x);
			current_dof_points[i*2] = v[0];
			current_dof_points[i*2+1] = v[1];
		}
	}

	/// Assign the solid displacement with the velocity of fluid.
	void fluid_to_solid(Function &fluid, Function &solid)
	{
		/// calculate global dof coordinates and dofs.
		std::vector<double> values(current_dof_points.size());
		fluid_to_solid_raw(fluid, values, current_dof_points);
		solid.vector()->set_local(values);
	}

	void fluid_to_solid_raw(const Function &fluid, 
							std::vector<double> &solid_values,
							std::vector<double> &solid_coordinates)
	{
		/// smart shortcut
		auto value_size   = fluid.value_size();

		/// iterate every solid_values coordinate.
		for (size_t i = 0; i < solid_values.size() / value_size; i++)
		{
			Array<double> x(2);
			Array<double> v(2);
			x[0] = solid_coordinates[2*i];
			x[1] = solid_coordinates[2*i+1];
			fluid.eval(v, x);
			for (size_t j = 0; j < value_size; j++)
				solid_values[i * value_size + j] = v[j];
		}
	}

	/// Assign the solid displacement with the velocity of fluid.
	void solid_to_fluid(Function &fluid, Function &solid)
	{
		/// calculate global dof coordinates and dofs of solid.
		std::vector<double> solid_values;
		for (size_t i = 0; i < reference_gauss_points.size()/2; ++i){
			Array<double> x(2);
			Array<double> v(2);
			x[0] = reference_gauss_points[2*i];
			x[1] = reference_gauss_points[2*i+1];
			solid.eval(v, x);
			solid_values.push_back(v[0]);
			solid_values.push_back(v[1]);
		}
		auto fluid_values = solid_to_fluid_raw(fluid, solid_values, current_gauss_points, weights);

		/// assemble fluid_values into a function.
		auto local_size = fluid.vector()->local_size();
		auto offset = fluid.vector()->local_range().first;
		std::vector<double> local_values(local_size);
		for (size_t i = 0; i < local_size; ++i)
		{
			local_values[i] = fluid_values[i + offset];
		}

		fluid.vector()->set_local(local_values);
		fluid.vector()->apply("insert");

	}

	std::vector<double> solid_to_fluid_raw(
		Function &fluid,
		std::vector<double> &solid_values,
		std::vector<double> &solid_coordinates,
		std::vector<double> &weights)
	{
		/// smart shortcut
		auto rank = dolfin::MPI::rank(fluid.function_space()->mesh()->mpi_comm());
		auto mesh = fluid.function_space()->mesh();		// pointer to a mesh
		auto dofmap = fluid.function_space()->dofmap(); // pointer to a dofmap

		/// get the element of function space
		auto element = fluid.function_space()->element();
		auto value_size = fluid.value_size();
		auto global_fluid_size = fluid.function_space()->dim();

		/// Get local to global dofmap
		std::vector<size_t> local_to_global;
		dofmap->tabulate_local_to_global_dofs(local_to_global);

		/// initial local fluid values.
		std::vector<double> local_fluid_values(global_fluid_size);

		/// iterate every solid_values coordinate.
		for (size_t i = 0; i < solid_values.size() / value_size; i++)
		{

			/// get indices of adjacent cells on fluid mesh.
			Point solid_point(solid_coordinates[2 * i], solid_coordinates[2 * i + 1]);
			auto adjacents = fluid_mesh->get_adjacents(solid_point);

			/// iterate adjacent cells and collect element nodes in these cells.
			/// it has nothing to do with cell type.
			std::map<size_t, double> indices_to_delta;
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

				/// iterate node coordinates of the cell.
				for (size_t k = 0; k < cell_dofmap.size() / value_size; k++)
				{
					Point cell_point(coordinates[k][0], coordinates[k][1]);
					double param = delta(solid_point, cell_point);
					if (cell_dofmap[k] < fluid.vector()->local_size() && param > 0.0)
					{
						indices_to_delta[cell_dofmap[k]] = param;
					}
				}
			}

			/// delta distribution.
			for (auto it = indices_to_delta.begin(); it != indices_to_delta.end(); it++)
			{
				for (size_t l = 0; l < value_size; l++)
				{
					local_fluid_values[local_to_global[it->first + l]] += solid_values[i * value_size + l] * it->second * weights[i];
				}
			}
		}
		/// f.vector()->add_local(data, dofs);
		//////////////////  TODO : this part can use MPI_reduce directly  //////////////////////
		std::vector<double> fluid_values(global_fluid_size);
		std::vector<std::vector<double>> mpi_collect(dolfin::MPI::size(mesh->mpi_comm()));
		dolfin::MPI::all_gather(mesh->mpi_comm(), local_fluid_values, mpi_collect);
		for (size_t i = 0; i < fluid_values.size(); i++)
		{
			for (size_t j = 0; j < mpi_collect.size(); j++)
			{
				fluid_values[i] += mpi_collect[j][i];
			}
		}
		return fluid_values;
	}

	/////////////////////////////////////////////
	//  thses methods must not be modified!!  ///
	/////////////////////////////////////////////
	double phi2(double r)
	{
		double phi;
		r = fabs(r);
		if (r<1 && r>=0){
			return 0.125*(3-2*r+sqrt(1+4*r-4*r*r));
		}
		if (r<2 && r>=1){
			return 0.125*(5-2*r-sqrt(-7+12*r-4*r*r));
		}
		return 0;
	}
	double phi1(double r)
	{
		r = fabs(r);
		if (r >= 2)
			return 0;
		else
			return 0.25 * (1 + cos(FENICS_PI * r * 0.5));
	}
	double delta(Point p0, Point p1, double h = 0.0625)
	{
		double ret = 1.0;
		/// 如果更改h的大小，fluid_mesh的bandwidth需要重新设置。
   		h = 0.5*side_lengths[0];
		for (unsigned i = 0; i < 2; ++i)
		{
			double dx = p0.coordinates()[i] - p1.coordinates()[i];
			ret *= 1. / h * phi2(dx / h);
		}
		return ret;
	}
	/////////////////////////////////////////////
	//  thses methods must not be modified!!  ///
	/////////////////////////////////////////////
};

namespace py = pybind11;
PYBIND11_MODULE(IBInterpolation, m)
{
    py::class_<IBInterpolation>(m, "IBInterpolation")
        .def(py::init<std::shared_ptr<IBMesh>, std::shared_ptr<Mesh>, std::shared_ptr<Function>>())
		.def("solid_to_fluid", &IBInterpolation::solid_to_fluid)
		.def("fluid_to_solid", &IBInterpolation::fluid_to_solid)
		.def("evaluate_current_points", &IBInterpolation::evaluate_current_points)
		;
}