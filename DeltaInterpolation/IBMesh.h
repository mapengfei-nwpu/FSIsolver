#ifndef _IBMesh_H_
#define _IBMesh_H_
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <dolfin.h>
using namespace dolfin;

class IBMesh
{
public:
	/// TODO : Add 3D version.
	/// TODO : Add hexahedron and tethedron.
	IBMesh(std::array<dolfin::Point, 2> points, std::vector<size_t> dims)//, CellType::Type cell_type = CellType::Type::triangle)
	{

		// toplogy dimesion
		// 二维情况下dims[2]等于零。
		top_dim = dims[2] == 0 ? 2 : 3;		

		// generate mesh
		// DIMENSION
		mesh_ptr = std::make_shared<Mesh>(
					 RectangleMesh::create(points, {dims[0], dims[1]}, CellType::Type::triangle, "left")
					);

		mpi_rank = dolfin::MPI::rank(mesh_ptr->mpi_comm());
		
		// check again.
		if (top_dim != 2 && top_dim != 3)
			dolfin_error("the size of dims must be 2 & 3.", ".", ".");

		nx = dims[0];
		ny = dims[1];
		nz = dims[2];

		/// TODO : check x1 > x0, y1>y0, z1>z0
		/// TODO : check x1-x0 > DOLFIN_EPS...
		x0 = points[0].x();
		x1 = points[1].x();
		y0 = points[0].y();
		y1 = points[1].y();
		z0 = points[0].z();
		z1 = points[1].z();

		index_mesh();
	}

	// global index and cell center
	std::array<size_t, 2> map(size_t i)
	{
		return global_map[i];
	}

	std::shared_ptr<Mesh> mesh()
	{
		return mesh_ptr;
	}
	/// 设置插值带宽
	void set_bandwidth(int b){
		bandwidth = b;
	}

	// point -> global index of adjacent cells -> local index of adjacent cells
	std::vector<size_t> get_adjacents(Point point)
	{
		/// only finished for 2d occassion.
		std::vector<size_t> adjacents;
		std::vector<size_t> a;
		std::vector<size_t> b;

		/// find all probable global adjacents.
		size_t global_index = hash(point);
		for (int i = -bandwidth; i <= bandwidth; i++){
			for (int j = -bandwidth; j <= bandwidth; j++){
				a.push_back(global_index + i * nx + j);
			}
		}
		for (size_t i = 0; i < a.size(); i++)
		{
			b.push_back(a[i] * 2);
			b.push_back(a[i] * 2 + 1);
		}

		/// select valid local cells.
		for (size_t i = 0; i < b.size(); i++)
		{
			if (b[i] >= 0 && b[i] < global_map.size() && map(b[i])[0] == mpi_rank)
			{
				adjacents.push_back(map(b[i])[1]);
			}
		}
		return adjacents;
	}

	// 网格单元边长
	std::vector<double> side_length()
	{
		std::vector<double> side_lengths;
		side_lengths.push_back((x1 - x0) / nx);
		side_lengths.push_back((y1 - y0) / ny);
		side_lengths.push_back((z1 - z0) / nz);
		return side_lengths;
	}

	void index_mesh()
	{
		// The local map local to global
		std::vector<size_t> local_map;
		for (dolfin::CellIterator e(*mesh_ptr); !e.end(); ++e)
		{
			auto center = e->midpoint();
			local_map.push_back(e->global_index());
			local_map.push_back(mpi_rank);
			local_map.push_back(e->index());
		}
		// send local map to every peocess.
		std::vector<std::vector<size_t>> mpi_collect(dolfin::MPI::size(mesh_ptr->mpi_comm()));
		dolfin::MPI::all_gather(mesh_ptr->mpi_comm(), local_map, mpi_collect);
		// alloc memory for global map.
		auto num_cell_global = mesh_ptr->num_entities_global(top_dim);
		global_map.resize(num_cell_global);
		for (auto iter = mpi_collect.cbegin(); iter != mpi_collect.cend(); iter++)
		{
			for (auto jter = iter->begin(); jter != iter->cend();)
			{
				size_t cell_index = *jter;
				jter++;
				global_map[cell_index][0] = *jter; /// mpi_rank;size_t
				jter++;
				global_map[cell_index][1] = *jter; /// cell local index
				jter++;
			}
		}
	}

	size_t hash(const dolfin::Point &point) const
	{
		double x = point.x();
		double y = point.y();
		double z = point.z();

		double dx = (x1 - x0) / static_cast<double>(nx);
		double dy = (y1 - y0) / static_cast<double>(ny);
		double dz = (z1 - z0) / static_cast<double>(nz);

		size_t i = static_cast<size_t>((x - x0) / dx);
		size_t j = static_cast<size_t>((y - y0) / dy);
		size_t k = 0;

		if (top_dim == 3)
			k = static_cast<size_t>((z - z0) / dz);

		return k * nx * ny + j * nx + i;
	}

	size_t valid_global_index(size_t global_index){
		if (global_map.size() > global_index){
			if (global_map[global_index][0] == mpi_rank){
				return global_map[global_index][1];
			}
		} else {
			return static_cast<size_t>(-1);
		}
	}

	bool find_cell(const Point &point, Cell &cell)
	{
    	auto index_1 = valid_global_index(2 * hash(point));
		auto index_2 = valid_global_index(2 * hash(point)+1);

		/// if index is not inside local mesh, assign it with 0.
		index_1 = mesh_ptr->num_entities(top_dim) > index_1 ? index_1 : 0;
		index_2 = mesh_ptr->num_entities(top_dim) > index_2 ? index_2 : 0;

		Cell cell_1(*mesh_ptr, index_1);
    	Cell cell_2(*mesh_ptr, index_2);

		cell = cell_1.contains(point) ? cell_1 : cell_2;
		
		if (cell.contains(point)){
			return true;
		} else {
			return false;
		}
	}
private:
	double x0, x1, y0, y1, z0, z1;
	size_t nx, ny, nz;
	size_t top_dim;
	size_t mpi_rank;
	int bandwidth = 1;
	// The map of global index to hash index for cells.
	std::vector<std::array<size_t, 2>> global_map;
	std::shared_ptr<Mesh> mesh_ptr;
};

namespace py = pybind11;
PYBIND11_MODULE(IBMesh, m)
{
    py::class_<IBMesh, std::shared_ptr<IBMesh>>(m, "IBMesh")
        .def(py::init<std::array<dolfin::Point, 2>, std::vector<size_t>>())
        .def("find_cell",&IBMesh::find_cell)
		.def("mesh",&IBMesh::mesh)
		.def("hash",&IBMesh::hash)
		.def("map",&IBMesh::map)
		.def("get_adjacents",&IBMesh::get_adjacents)
		.def("side_length",&IBMesh::side_length)
		;
}

#endif