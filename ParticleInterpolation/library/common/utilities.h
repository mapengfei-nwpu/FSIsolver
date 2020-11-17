#ifndef UTILITIES_H
#define UTILITIES_H

#include <dolfin.h>

/// gather all data on all processors
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

/// type conversion between std::vector<double> and std::vector<float>.
template <typename T1, typename T2>
void vectorTypeConvert(const std::vector<T1> &from, std::vector<T2> &to)
{
    to.resize(from.size());
    for (size_t i = 0; i < from.size(); i++)
    {
        to[i] = static_cast<T2>(from[i]);
    }
}

# ifndef VECTOR_TXT_H
# define VECTOR_TXT_H
#include<iostream>
#include<fstream>
#include<vector>
#include<string>

//////////////////////////////////////
// How to use?
//
// input a vector from a txt file:
// input(file, data);
//
// output a txt file from a vector:
// output(file, data);
// 
/////////////////////////////////////
template <typename T>
void input(std::string filename, std::vector<T> &data){
    if (data.size() != 0) std::cout << "the destination isn't empty!" << std::endl;
    std::cout << "Reading a vector from \"" << filename <<"\"."<< std::endl;
    std::ifstream  ifs(filename);
    if(!ifs.is_open()) {
        std::cout << "Failed to open \"" << filename <<"\"."<< std::endl;
        return;
    }
    while(ifs.eof()==0){
        T temp;
        ifs >> temp;
        data.push_back(temp);
    }
    data.pop_back();
    ifs.close();
}

template <typename T>
void output(std::string filename, std::vector<T> &data){
    std::cout << "Writing a vector to \"" << filename <<"\"."<< std::endl; 
    std::ofstream  ofs(filename, std::ios::out);
    for(size_t i = 0; i < data.size(); i++)
        ofs << data[i] << std::endl;
    ofs.close();
}

# endif



#endif
