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


/// separate a vector into several vectors.
/// [a1,b1,c1...,a2,b2,c2,...] <-> [[a1,a2...],[b1,b2...]...]

template <typename T>
void separate_vector(std::vector<std::vector<T>> &separate, std::vector<T> &connect, size_t num_vectors){
    
    // assert(separate.size() > 0);
    // assert(connect.size() % num_vectors == 0);
    size_t num_elements = connect.size() % num_vectors;

    separate.resize(num_vectors);
    for (size_t i = 0; i < num_vectors; i++){
        separate[i].resize(num_elements);
    }

    for (size_t i = 0; i < num_elements; i++){
        for (size_t j = 0; j < num_vectors; j++){
            separate[j][i] = connect[num_vectors*i+j];
        }
    }
}

template <typename T>
void connect_vectors(std::vector<std::vector<T>> &separate, std::vector<T> &connect){
    
    /// assert(separate.size() > 0);
    /// assert(connect.size() == 0);
    size_t num_vectors = separate.size();
    size_t num_elements = separate[0].size();

    for (size_t i = 0; i < separate.size(); i++){
        /// assert(separate[i].size() == num_elements);
    }

    for (size_t i = 0; i < num_elements; i++){
        for (size_t j = 0; j < num_vectors; j++){
            connect.append(separate[j][i]);
        }
    }
}



# endif

