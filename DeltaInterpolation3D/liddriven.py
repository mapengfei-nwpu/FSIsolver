# cp ../liddriven.py . && mpirun -np 8 python3 liddriven.py
from dolfin import *
from mshr import *
from IB import *
points = [Point(0, 0, 0), Point(1, 1, 1)]
seperations = [32, 32, 32]

regular_mesh = IBMesh(points, seperations)
fluid_mesh = regular_mesh.mesh()
solid_mesh = generate_mesh(Sphere(Point(0.6,0.5,0.5), 0.2),15)

# Define function spaces (P2-P1) for fluid and Vs for solid
P1 = FunctionSpace(fluid_mesh, "Lagrange", 1)
P2 = VectorFunctionSpace(fluid_mesh, "Lagrange", 2)
Vs = VectorFunctionSpace(solid_mesh, "Lagrange", 2)

# Define trial and test functions
u_v = TrialFunction(P2)
u_s = TrialFunction(P1)
v_v = TestFunction(P2)
v_s = TestFunction(P1)
us = TrialFunction(Vs)
vs = TestFunction(Vs)

# Set parameter values
dt = 0.125/32.0
T = 10
nu = 0.01
nu_s = 0.1

# Define boundary conditions
noslip  = DirichletBC(P2, (0, 0, 0), "near(x[0], 1) || near(x[0], 0) || near(x[1], 0)")
upflow  = DirichletBC(P2, (1, 0, 0), "near(x[1], 1)")
pinpoint = DirichletBC(P1, 0, "near(x[0],0) && near(x[1],0) && near(x[2],0)", "pointwise")
bcu = [noslip, upflow]
bcp = [pinpoint]

# Create functions for fluid
u = Function(P2)
u_ = Function(P2)
p = Function(P1)
f = Function(P2)

# Create functions for solid
velocity = Function(Vs)
disp = Function(Vs)
force = Function(Vs)
disp.interpolate(Expression(("x[0]","x[1]","x[2]"),degree=2))
IB = DeltaInterpolation(regular_mesh, solid_mesh, disp._cpp_object)

# Define coefficients
k = Constant(dt)

# Tentative velocity step
F1 = (1/k)*inner(u_v - u, v_v)*dx + inner(grad(u)*u, v_v)*dx + nu*inner(grad(u_v), grad(v_v))*dx - inner(f,v_v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(u_s), grad(v_s))*dx
L2 = -(1/k)*div(u_)*v_s*dx

# Velocity update
a3 = inner(u_v, v_v)*dx
L3 = inner(u_, v_v)*dx - k*inner(grad(p), v_v)*dx

# Define variational problem for solid
F4 = nu_s*inner(grad(disp), grad(vs))*dx + inner(us, vs)*dx
a4 = lhs(F4)
L4 = rhs(F4)

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)
A4 = assemble(a4)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")
dfile = File("results/disp.pvd")
ffile = File("results/force.pvd")
# Time-stepping
t = dt
while t < T + DOLFIN_EPS:
    # step 1. calculate velocity and pressure
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u_.vector(), b1, "bicgstab", prec)
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p.vector(), b2, "bicgstab", prec)
    b3 = assemble(L3)
    [bc.apply(A3, b3) for bc in bcu]
    solve(A3, u.vector(), b3, "bicgstab", prec)
    # step 2. interpolate velocity from fluid to solid
    IB.fluid_to_solid(u._cpp_object, velocity._cpp_object)
    # step 3. calculate disp for solid and update current gauss points and dof points
    disp.vector()[:] = velocity.vector()[:]*dt + disp.vector()[:]
    IB.evaluate_current_points(disp._cpp_object)
    # step 4. calculate body force.
    b4 = assemble(L4)
    solve(A4, force.vector(), b4)
    # step 5. interpolate force from solid to fluid
    IB.solid_to_fluid(f._cpp_object, force._cpp_object)
    # step 6. update variables and save to file.
    # Update time and step
    t += dt
    # Save to file
    ufile << u
    pfile << p
    ffile << f
    dfile << disp
    print("t:",t)






