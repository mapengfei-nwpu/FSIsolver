#ifndef __POLYNOMIALINTERPOLATION_H__
#define __POLYNOMIALINTERPOLATION_H__
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <assert.h>



class PolynomialInterpolation {
public:
	PolynomialInterpolation(){
		ref_basis_matrix();
	}
private:
	void linear_transformation(double* p, double *F, double* b);

	void point_local_to_ref(double* point_out, double* point_in, double *F, double* b);

public:
	void evaluate_basis_at_point(double* point, double* basis);

	void transform_points(double* points_ref, double* points, double* coordinates_dof, size_t num_points);
	
	void evaluate_function_at_points(double* dof, double* dof_params, size_t num_dof, double* points_ref, size_t num_points, size_t value_size, double* results);

	void evaluate_function(size_t num_cells, size_t num_gauss, size_t value_size, size_t num_dofs,
		double* coordinates, double* dofs, double* gauss_points, double* results);

private:
	bool useCuda = true;
	double second_tetrahedron_dof_points[10][3] ={
		{0.0, 0.0 ,0.0},
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
		{0.0, 0.5, 0.5},
		{0.5, 0.0, 0.5},
		{0.5, 0.5, 0.0}, 
		{0.0, 0.0, 0.5},
		{0.0, 0.5, 0.0},
		{0.5, 0.0, 0.0}
	};
	double G_inv[10*10] = {0};

private:
	void ref_basis_matrix();
};

#endif