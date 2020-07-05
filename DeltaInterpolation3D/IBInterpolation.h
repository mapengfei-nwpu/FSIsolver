#ifndef _IBINTERPOLATION_H_
#define _IBINTERPOLATION_H_
#include <dolfin.h>
#include <dolfin/geometry/SimplexQuadrature.h>
#include <numeric>
#include <ctime>
#include "IBMesh.h"
#include "LocalPolynomial.h"
namespace dolfin
{
/// the usage of DeltaInterpolation:
/// DeltaInterpolation di(ib_mesh);
/// di.fluid_to_soild(fluid,solid);
/// di.solid_to_fluid(fluid,solid);

/// the method for interpolation can be modified:
/// bandwidth is the bandwidth,
/// delta is the delta,
/// they can be modified.
class DeltaInterpolation
{
public:
	DeltaInterpolation(std::shared_ptr<IBMesh> fluid_mesh, 
									 std::shared_ptr<Mesh> solid_mesh,
					                 std::shared_ptr<Function> solid) ;

	void fluid_to_solid(Function &fluid, Function &solid);

	void solid_to_fluid(Function &fluid, Function &solid);

	void evaluate_current_points(std::shared_ptr<const Function> disp);

private:
	int bandwidth = 2;
	std::shared_ptr<IBMesh> fluid_mesh;
	std::shared_ptr<const Mesh> solid_mesh;
	std::vector<double> side_lengths;
public:
	/// Define solid coordinates of gauss points and their weights.
	std::vector<double> gauss_points_current;
	std::vector<double> gauss_points_reference;
	std::vector<double> weights;
	std::vector<size_t> gauss_points_to_cell;

	/// Define solid coordinates of dofs.
	std::vector<double> dof_points_current;

	/// Time cost on my_mpi_gather
	std::clock_t time_mpi = 0;
private:
	/// mid step for solid to fluid interpolation.
	void solid_to_fluid_raw(
		Function &fluid,
		std::vector<double> &solid_values,
		std::vector<double> &solid_coordinates,
		std::vector<double> &weights);
	/// update coordinates.
	void update(
			std::shared_ptr<const Function> disp,
			const std::vector<size_t> &points_to_cell,
			const std::vector<double> &reference_points,
				  std::vector<double> &current_points
			);

	/////////////////////////////////////////////
	//  thses methods must not be modified!!  ///
	/////////////////////////////////////////////
	double phi(double r)
	{
		r = fabs(r);
		if (r > 2)
			return 0;
		else
			return 0.25 * (1 + cos(FENICS_PI * r * 0.5));
	}
	double delta(const double* p0, const double* p1)
	{
		double ret = 1.0;
		for (unsigned i = 0; i < 3; ++i)
		{
			double dx = p0[i] - p1[i];
			ret *= 1. / side_lengths[i] * phi(dx / side_lengths[i]);
		}
		return ret;
	}
	/////////////////////////////////////////////
	//  thses methods must not be modified!!  ///
	/////////////////////////////////////////////
};
} // namespace dolfin
#endif