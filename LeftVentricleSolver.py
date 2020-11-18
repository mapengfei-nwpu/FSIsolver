from fenics import *

from LeftVentricleMesh     import mesh          as solid_mesh
from LeftVentricleMesh     import mesh_function as solid_boundary
from PassiveLeftVentricle  import first_PK_stress

from NavierStokesSolver    import ProjectSolver as FluidSolver 
from ParticleInterpolation import ParticleInterpolation

# Set parameter for the fluid.
n_mesh = 32
dt = 0.125/n_mesh
T = 10
nu = 4
points = [Point(-10, -10,-20), Point(10, 10, 5)]
fluid_mesh = BoxMesh(points[0], points[1], n_mesh, n_mesh, n_mesh)

# Define function spaces for Navier-Stokes equations
V = VectorFunctionSpace(fluid_mesh, "Lagrange", 2)
Q = FunctionSpace(fluid_mesh, "Lagrange", 2)

# Define function space for left ventricle
Vs = VectorFunctionSpace(solid_mesh, "Lagrange", 2)

# Define trial and test functions for solid
us = TrialFunction(Vs)
vs = TestFunction(Vs)

# Define boundary conditions for the fluid
noslip = DirichletBC(V, (0, 0, 0), "near(x[0],1) || near(x[0],0) || near(x[1],0)")
upflow = DirichletBC(V, (1, 0, 0), "near(x[1],1)")
pinpoint = DirichletBC(Q, 0, "near(x[0],0) && near(x[1],0) && near(x[2],0)", "pointwise")
bcu = []
bcp = []

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

# Define the constutive law for the left ventricle
F = inner(first_PK_stress(disp), grad(vs))*dx + inner(us, vs)*dx
a = lhs(F)
L = rhs(F)
A = assemble(a)

# solver start.
t = dt
while t < T + DOLFIN_EPS:
    # step 1. calculate velocity and pressure
    u1, p1 = navier_stokes_solver.solve(u0, p0, bcu, bcp)
    u0.assign(u1)
    p0.assign(p1)
    # step 2. interpolate velocity from fluid to solid
    IB.fluid_to_solid(u0, velocity)
    # step 3. calculate disp for solid and update current gauss points and dof points
    disp.vector()[:] = velocity.vector()[:]*dt + disp.vector()[:]
    IB.evaluate_current_points(disp)
    # step 4. calculate body force.
    b = assemble(L)
    solve(A, force.vector(), b, 'cg', 'sor' )
    # step 5. interpolate force from solid to fluid
    IB.solid_to_fluid(f, force)
    # step 6. update variables and save to file.
    t += dt
    print(t)








