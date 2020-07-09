from NavierStokesSolver import *
from DeltaInterpolation3D import *
from dolfin import *
from mshr import *

# Define fluid solver
FluidSolver = IPCSSolver

# Set parameter values
n_mesh = 32
dt = 0.125/n_mesh
T = 10
nu = 0.01
nu_s = 0.1

# Define fluid_mesh
points = [Point(0, 0, 0), Point(1, 1, 1)]
seperations = [n_mesh, n_mesh, n_mesh]
regular_mesh = IBMesh(points, seperations)
fluid_mesh = regular_mesh.mesh()
solid_mesh = generate_mesh(Sphere(Point(0.6, 0.5, 0.5), 0.2), 15)

# Define function spaces W for Navier-Stokes equations
V = VectorFunctionSpace(fluid_mesh, "Lagrange", 2)
Q = FunctionSpace(fluid_mesh, "Lagrange", 2)

# Define function spaces W for solid
Vs = VectorFunctionSpace(solid_mesh, "Lagrange", 2)

# Define trial and test functions for solid
us = TrialFunction(Vs)
vs = TestFunction(Vs)

# Define boundary conditions
noslip = DirichletBC(V, (0, 0, 0), "near(x[0],1) || near(x[0],0) || near(x[1],0)")
upflow = DirichletBC(V, (1, 0, 0), "near(x[1],1)")
pinpoint = DirichletBC(Q, 0, "near(x[0],0) && near(x[1],0) && near(x[2],0)", "pointwise")
bcu = [noslip, upflow]
bcp = [pinpoint]
# bcp = []

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
IB = DeltaInterpolation(regular_mesh, solid_mesh, disp._cpp_object)
IB.evaluate_current_points(disp._cpp_object)
# -inv(grad(disp)).T
# Define variational problem for solid
F2 = nu_s*inner(grad(disp)-inv(grad(disp)).T, grad(vs))*dx + inner(us, vs)*dx
a2 = lhs(F2)
L2 = rhs(F2)
A2 = assemble(a2)

# Output Directory name
directory = "results3D/meshsize_" + str(n_mesh) + "/nu_s_" + str(nu_s)
print("output directory : ", directory)

# Create files for storing solution
ufile = File(directory + "/velocity.pvd")
pfile = File(directory + "/pressure.pvd")
ffile = File(directory + "/force.pvd")

dfile = File(directory + "/disp.pvd")
ufile2 = File(directory + "/velocity2.pvd")
ffile2 = File(directory + "/force2.pvd")

t = dt
while t < T + DOLFIN_EPS:
    # step 1. calculate velocity and pressure
    u1, p1 = navier_stokes_solver.solve(u0, p0, bcu, bcp)
    u0.assign(u1)
    p0.assign(p1)
    # step 2. interpolate velocity from fluid to solid
    IB.fluid_to_solid(u0._cpp_object, velocity._cpp_object)
    # step 3. calculate disp for solid and update current gauss points and dof points
    disp.vector()[:] = velocity.vector()[:]*dt + disp.vector()[:]
    IB.evaluate_current_points(disp._cpp_object)
    # step 4. calculate body force.
    b2 = assemble(L2)
    solve(A2, force.vector(), b2, 'cg', 'sor' )
    # step 5. interpolate force from solid to fluid
    IB.solid_to_fluid(f._cpp_object, force._cpp_object)
    # step 6. update variables and save to file.
    ufile2 << velocity
    ufile << u0
    pfile << p0
    ffile << f
    ffile2 << force
    dfile << disp
    t += dt
    print(t)
