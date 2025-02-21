#include <iostream>
#include <dolfin.h>
#include "IBMesh.h"
using namespace dolfin;

IBMesh::IBMesh(std::array<Point, 2> points,
			   std::vector<std::size_t> dims)
{
	nx = dims[0];
	ny = dims[1];
	nz = dims[2];

	x0 = points[0].x();
	x1 = points[1].x();
	y0 = points[0].y();
	y1 = points[1].y();
	z0 = points[0].z();
	z1 = points[1].z();

	// generate mesh
	mesh_ptr = std::make_shared<Mesh>(
				 BoxMesh::create(points, {nx, ny, nz}, CellType::Type::hexahedron)
				);
	std::cout << "mesh is created!" << std::endl;

	top_dim = mesh_ptr->topology().dim();
	mpi_rank = MPI::rank(mesh_ptr->mpi_comm());

	/// set up global map of this mesh
	index_mesh();
}

std::vector<std::size_t> IBMesh::get_adjacents(const Point& point, int bandwidth)
{

	/// initial neighbours
	std::vector<std::size_t> adjacents;

	/// test if the point is inside the box
	if (!(point.x() < x1 && point.x() > x0 &&
		  point.y() < y1 && point.y() > y0 &&
		  point.z() < z1 && point.z() > z0))
	{
		std::cout << "searching adjacents." << std::endl;
		std::cout << "the point is not inside the box." << std::endl;
		return adjacents;
	}

	/// set width of adjacent cells
	std::size_t global_index = hash(point);

	/// search adjacent cells in current mpi processor
	/// TODO : try to do it with inner method.
	///        topology might be able to map global index to local index.
	for (int iz = -bandwidth; iz <= bandwidth; iz++)
	{
		for (int iy = -bandwidth; iy <= bandwidth; iy++)
		{
			for (int ix = -bandwidth; ix <= bandwidth; ix++)
			{
				auto neighbour = static_cast<int>(global_index) + iz * nx * ny + iy * nx + ix;
				/// excluding unsatisfied cell index.
				if (neighbour >= 0 && neighbour < global_map.size())
				{
					if (global_map[neighbour][0] == mpi_rank)
					{
						adjacents.push_back(global_map[neighbour][1]);
					}
				}
				/// end excluding
			}
		}
	}
	/// end neighbour searching
	return adjacents;
}


// global index to local index
std::array<std::size_t, 2> IBMesh::map(std::size_t i)
{
	return global_map[i];
}

// TODO : will this function create another copy of mesh?
std::shared_ptr<Mesh> IBMesh::mesh()
{
	return mesh_ptr;
}

// return the length of every side of the mesh.
std::vector<double> IBMesh::cell_length()
{
	std::vector<double> cell_lengths;
	cell_lengths.push_back((x1 - x0) / nx);
	cell_lengths.push_back((y1 - y0) / ny);
	cell_lengths.push_back((z1 - z0) / nz);
	return cell_lengths;
}

// given a global cell index i, global_map return a tuple (a,b),
// where "a" is the local index and "b" is the mpi processor of
// this cell.
void IBMesh::index_mesh()
{
	// local_map is vector whith 3*cell_num entries.
	// It contains the global index, local index, mpi rank
	std::vector<std::size_t> local_map;
	for (CellIterator e(*mesh_ptr); !e.end(); ++e)
	{
		local_map.push_back(e->global_index());
		local_map.push_back(mpi_rank);
		local_map.push_back(e->index());
	}

	// TODO : There is no need to have a copy on every processor.
	//        The type of global_map should be std::map<std::size_t,std::size_t>.
	//        It is a bargain between searching time and memory cost.

	// collect local map on every peocess.
	std::vector<std::vector<std::size_t>> mpi_collect(MPI::size(mesh_ptr->mpi_comm()));
	MPI::all_gather(mesh_ptr->mpi_comm(), local_map, mpi_collect);

	// alloc memory for global map.
	// global map just resize the collected local index.
	auto num_cell_global = mesh_ptr->num_entities_global(3);
	global_map.resize(num_cell_global);
	for (auto iter = mpi_collect.cbegin(); iter != mpi_collect.cend(); iter++)
	{
		for (auto jter = iter->begin(); jter != iter->cend();)
		{
			std::size_t cell_index = *jter;
			jter++;
			global_map[cell_index][0] = *jter; /// mpi_rank
			jter++;
			global_map[cell_index][1] = *jter; /// cell local index
			jter++;
		}
	}
}

// Every point has a unique cell index.
// It is consistent with the generation of the mesh.
std::size_t IBMesh::hash(const Point& point)
{
	double x = point.x();
	double y = point.y();
	double z = point.z();

	double dx = (x1 - x0) / static_cast<double>(nx);
	double dy = (y1 - y0) / static_cast<double>(ny);
	double dz = (z1 - z0) / static_cast<double>(nz);

	std::size_t i = static_cast<std::size_t>((x - x0) / dx);
	std::size_t j = static_cast<std::size_t>((y - y0) / dy);
	std::size_t k = static_cast<std::size_t>((z - z0) / dz);

	return k * nx * ny + j * nx + i;
}