from fenics import *
from loguru import logger
from LeftVentricleMesh     import mesh          as solid_mesh
from LeftVentricleMesh     import mesh_function as solid_boundary
from PassiveLeftVentricle  import first_PK_stress
from LeftVentricleBoundaryCondition import apply_boundary_conditions

from NavierStokesSolver    import ProjectSolver as FluidSolver 
from ParticleInterpolation import ParticleInterpolation
from PeriodicalBoundary    import periodic_boundary
# Set parameter for the fluid.
n_mesh = 32
dt = 1e-4
T = 10
nu = 0.04
points = [Point(-2.0, -2.0,-3.0), Point(2.0, 2.0, 1.0)]
fluid_mesh = BoxMesh(points[0], points[1], n_mesh, n_mesh, n_mesh)

# the markers of the boundary
marker_base = 1
marker_wall = 2

# boundary conditions
bcp = []
bcu = []

# Define function spaces for Navier-Stokes equations
V = VectorFunctionSpace(fluid_mesh, "Lagrange", 2)#, constrained_domain=periodic_boundary)
Q = FunctionSpace(fluid_mesh, "Lagrange", 2)#, constrained_domain=periodic_boundary)

# Define function space for left ventricle
Vs = VectorFunctionSpace(solid_mesh, "Lagrange", 2)

# Define trial and test functions for solid
us = TrialFunction(Vs)
vs = TestFunction(Vs)

# Create functions for fluid
u0 = Function(V)
p0 = Function(Q)
f = Function(V)

# Create functions for solid
velocity = Function(Vs)
disp = Function(Vs)
force = Function(Vs)
disp.interpolate(Expression(("x[0]", "x[1]", "x[2]"), degree=2))

# Define interpolation object and fluid solver object
navier_stokes_solver = FluidSolver(u0, p0, f, dt=dt, nu=nu)
IB = ParticleInterpolation(fluid_mesh, solid_mesh, disp)
IB.evaluate_current_points(disp._cpp_object)

F = inner(first_PK_stress(disp), grad(vs))*dx + inner(us, vs)*dx
a = lhs(F)
L = rhs(F)
A = assemble(a)

file_velocity = File("velocity.pvd")
file_velocity2 = File("velocity2.pvd")
file_force = File("force.pvd")
file_force2 = File("force2.pvd")
file_disp = File("disp.pvd")
file_disp2 = File("disp2.pvd")

# solver start.
t = dt
import time
while t < T + DOLFIN_EPS:
    # step 1. calculate velocity and pressure
    time_start=time.time()
    u1, p1 = navier_stokes_solver.solve(u0, p0, bcu, bcp)
    u0.assign(u1)
    p0.assign(p1)
    file_velocity << u1
    time_end=time.time()
    print('fluid solver : ',time_end-time_start,' second')
    # step 2. interpolate velocity from fluid to solid
    time_start=time.time()
    IB.fluid_to_solid(u0, velocity)
    # step 3. calculate disp for solid and update current gauss points and dof points
    disp.vector()[:] = velocity.vector()[:]*dt + disp.vector()[:]
    IB.evaluate_current_points(disp)
    time_end=time.time()
    print('interpolation from fluid to solid : ',time_end-time_start,' second')
    # step 4. calculate body force.
    time_start=time.time()
    b = assemble(L)
    solve(A, force.vector(), b, 'cg', 'sor' )
    time_end=time.time()
    print('solid solver : ',time_end-time_start,' second')
    # apply the boundary of the solid
    apply_boundary_conditions(disp,force,solid_boundary,marker_base,marker_wall)
    force.vector()[:] = force.vector()[:]*1e4
    file_force << force
    # step 5. interpolate force from solid to fluid
    time_start=time.time()
    IB.solid_to_fluid(f, force)
    time_end=time.time()
    print('interpolation from solid to fluid : ',time_end-time_start,' second')
    logger.info("interpolation from solid to fluid")
    file_force2 << f
    # step 6. update variables and save to file.
    file_disp << disp
    t += dt
    print(t)



#TODO: 1.Active stress tensor.
#TODO: 4.Unit conversion: mesh size, viscous coefficient, pressure.






