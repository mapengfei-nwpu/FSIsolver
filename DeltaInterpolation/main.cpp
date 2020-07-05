#include "IBMesh.h"
#include "IBInterpolation.h"

#include <dolfin.h>
#include "TentativeVelocity.h"
#include "PressureUpdate.h"
#include "VelocityUpdate.h"
#include "ElasticStructure.h"

using namespace dolfin;

// Define noslip domain
class NoslipDomain : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return near(x[0], 0) || near(x[0], 1.0) || near(x[1], 0);
  }
};

// Define inflow domain
class InflowDomain : public SubDomain
{
  bool inside(const Array<double> &x, bool on_boundary) const
  {
    return near(x[1], 1);
  }
};

// Define pressure boundary value at inflow
class InflowVelocity : public Expression
{
public:
  // Constructor
  InflowVelocity() : Expression(2) {}

  // Evaluate pressure at inflow
  void eval(Array<double> &values, const Array<double> &x) const
  {
    values[0] = 1.0;
    values[1] = 0.0;
  }
};

void my_move(Mesh &mesh, Function &displacement)
{
  std::vector<double> vertex_values;
  const std::size_t N = mesh.num_vertices();
  displacement.compute_vertex_values(vertex_values);

  // Move vertex coordinates
  MeshGeometry &geometry = mesh.geometry();
  std::vector<double> x(2);
  for (std::size_t i = 0; i < N; i++)
  {
    for (std::size_t j = 0; j < 2; j++)
      x[j] = geometry.x(i, j) + vertex_values[j * N + i];
    geometry.set(i, x.data());
  }
}

int main()
{
  // Create chanel mesh
  Point point0(0, 0, 0);
  Point point1(1.0, 1.0, 0);
  IBMesh ba({point0, point1}, {64, 64});
  DeltaInterplation interpolation(ba);

  // Create circle mesh
  auto circle = std::make_shared<Mesh>("./circle.xml.gz");
  auto U = std::make_shared<ElasticStructure::FunctionSpace>(circle);
  auto body_velocity = std::make_shared<Function>(U);
  auto body_force = std::make_shared<Function>(U);
  auto body_disp = std::make_shared<Function>(U);

  // Create function spaces
  auto V = std::make_shared<VelocityUpdate::FunctionSpace>(ba.mesh());
  auto Q = std::make_shared<PressureUpdate::FunctionSpace>(ba.mesh());

  // Set parameter values
  double dt = 0.01;
  double T = 100;

  // Define values for boundary conditions
  auto v_in = std::make_shared<InflowVelocity>();
  auto zero = std::make_shared<Constant>(0.0);
  auto zero_vector = std::make_shared<Constant>(0.0, 0.0);

  // Define subdomains for boundary conditions
  auto noslip_domain = std::make_shared<NoslipDomain>();
  auto inflow_domain = std::make_shared<InflowDomain>();

  // Define boundary conditions
  DirichletBC noslip(V, zero_vector, noslip_domain);
  DirichletBC inflow(V, v_in, inflow_domain);
  std::vector<DirichletBC *> bcu = {{&inflow, &noslip}};
  std::vector<DirichletBC *> bcp = {};

  // Create functions
  auto u_ = std::make_shared<Function>(V);
  auto u_n = std::make_shared<Function>(V);
  auto p_ = std::make_shared<Function>(Q);
  auto p_n = std::make_shared<Function>(Q);

  // Create coefficients
  auto k = std::make_shared<Constant>(dt);
  auto f = std::make_shared<Function>(V);
  auto fluid_force = std::make_shared<Function>(V);

  // Create forms
  TentativeVelocity::BilinearForm a1(V, V);
  TentativeVelocity::LinearForm L1(V);
  PressureUpdate::BilinearForm a2(Q, Q);
  PressureUpdate::LinearForm L2(Q);
  VelocityUpdate::BilinearForm a3(V, V);
  VelocityUpdate::LinearForm L3(V);
  ElasticStructure::BilinearForm a4(U, U);
  ElasticStructure::LinearForm L4(U);

  // Set coefficients
  a1.k = k;
  L1.k = k;
  L1.u_n = u_n;
  L1.p_n = p_n;
  L1.f = f;
  L2.k = k;
  L2.u_ = u_;
  L2.p_n = p_n;
  L3.k = k;
  L3.u_ = u_;
  L3.p_ = p_;
  L3.p_n = p_n;
  L4.u = body_disp;

  // Assemble matrices
  Matrix A1, A2, A3, A4;
  assemble(A1, a1);
  assemble(A2, a2);
  assemble(A3, a3);
  assemble(A4, a4);

  // Create vectors
  Vector b1, b2, b3, b4;

  // Create files for storing solution
  File ufile("results/velocity.pvd");
  File pfile("results/pressure.pvd");
  File ffile("results/force.pvd");
  File bfile("results/body.pvd");

  // Time-stepping
  double t = dt;
  while (t < T + DOLFIN_EPS)
  {
    // Compute tentative velocity step
    begin("Computing tentative velocity");
    assemble(b1, L1);
    for (std::size_t i = 0; i < bcu.size(); i++)
      bcu[i]->apply(A1, b1);
    solve(A1, *u_->vector(), b1, "bicgstab", "hypre_amg");
    end();

    // Pressure correction
    begin("Computing pressure correction");
    assemble(b2, L2);
    for (std::size_t i = 0; i < bcp.size(); i++)
    {
      bcp[i]->apply(A2, b2);
      bcp[i]->apply(*p_->vector());
    }
    solve(A2, *p_->vector(), b2, "bicgstab", "hypre_amg");
    end();

    // Velocity correction
    begin("Computing velocity correction");
    assemble(b3, L3);
    for (std::size_t i = 0; i < bcu.size(); i++)
      bcu[i]->apply(A3, b3);
    solve(A3, *u_->vector(), b3, "cg", "sor");
    end();

    // force calculation
    begin("Computing elastic force");
    auto temp_disp = std::make_shared<Function>(U);

    my_move(*circle, *body_disp);
    interpolation.fluid_to_solid(*u_, *body_velocity);
    *temp_disp = FunctionAXPY(body_disp, -1.0);
    my_move(*circle, *temp_disp);

    *temp_disp = FunctionAXPY(body_velocity, dt) + body_disp;
    *body_disp = *temp_disp;
    assemble(b4, L4);
    solve(A4, *body_force->vector(), b4);

    my_move(*circle, *body_disp);
    bfile << *body_force;
    interpolation.solid_to_fluid(*f, *body_force);
    *temp_disp = FunctionAXPY(body_disp, -1.0);
    my_move(*circle, *temp_disp);

    end();

    // Save to file
    ufile << *u_;
    pfile << *p_;
    ffile << *f;

    // Move to next time step
    *u_n = *u_;
    *p_n = *p_;
    t += dt;
    cout << "t = " << t << endl;
  }

  return 0;
}
