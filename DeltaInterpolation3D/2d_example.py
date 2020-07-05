from dolfin import *
from mshr import *

from IBInterpolation import *
from IBMesh import *

# import matplotlib.pyplot as plt

# Define fluid_mesh
points = [Point(0, 0, 0), Point(1, 1, 0)]
seperations = [64, 64, 0]
regular_mesh = IBMesh(points, seperations)
fluid_mesh = regular_mesh.mesh()
solid_mesh = generate_mesh(Circle(Point(0.6, 0.5), 0.2), 15)

# Define function spaces W for Navier-Stokes equations
P2 = VectorElement("Lagrange", fluid_mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", fluid_mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(fluid_mesh, TH)

# Define function spaces W for solid
Vs = VectorFunctionSpace(solid_mesh, "Lagrange", 2)

# Define trial and test functions for fluid
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Define trial and test functions for solid
us = TrialFunction(Vs)
vs = TestFunction(Vs)

# Set parameter values
dt = 0.125/64.0
T = 10
delta = 0.1
nu = 0.01
nu_s = 0.1

# Define boundary conditions
noslip  = DirichletBC(W.sub(0), (0, 0), "near(x[0], 1) || near(x[0], 0) || near(x[1], 0)")
upflow  = DirichletBC(W.sub(0), (1, 0), "near(x[1], 1)")
pinpoint = DirichletBC(W.sub(1), 0, "near(x[0],0) && near(x[1],0)","pointwise")
bcns = [noslip, upflow, pinpoint]
bcsav = [DirichletBC(W.sub(0), (0, 0), "on_boundary"),
        DirichletBC(W.sub(1), 0, "on_boundary", "pointwise")]

# Create functions for fluid
w1 = Function(W)
w2 = Function(W)
w = Function(W)
(un, pn) = w.split()
(f, _) = Function(W).split(True)

# Create functions for solid
velocity = Function(Vs)
disp = Function(Vs)
force = Function(Vs)
disp.interpolate(Expression(("x[0]","x[1]"),degree=2))
SS = Expression("S", degree=0, S=1)
def calculate_s(w1,w2,un):
    (u1,p1)=w1.split()
    (u2,p2)=w2.split()
    temp = 0.5*assemble(inner(un, un)*dx) + delta
    r = sqrt(temp)
    a2 = nu*assemble(inner(grad(u2), grad(u2))*dx) + 2/dt*temp
    a1 = 2*nu*assemble(inner(grad(u2), grad(u1))*dx) - 2 * \
        r/dt*sqrt(temp)-assemble(inner(f, u2)*dx)
    a0 = nu*assemble(inner(grad(u1), grad(u1))*dx) - assemble(inner(f, u1)*dx)
    s1 = (-a1+sqrt(a1*a1-4*a2*a0))/2/a2
    s2 = (-a1-sqrt(a1*a1-4*a2*a0))/2/a2
    # print(s1, s2)
    if max(s1, s2) > 0 and min(s1, s2) < 0:
        return max(s1, s2)
    if abs(s1-1) > abs(s2-1):
        return s2
    else:
        return s1

# Define interpolation object
IB = IBInterpolation(regular_mesh, solid_mesh, disp._cpp_object)

# Define variational problem
k = Constant(dt)
F1 = inner((u - un)/k, v)*dx + nu * inner(grad(u), grad(v)) * dx - div(v)*p*dx + q*div(u)*dx - inner(f,v)*dx
F3 = inner(u/k, v)*dx + inner(grad(un) * un, v)*dx + nu * inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx

a1 = lhs(F1)
L1 = rhs(F1)
A1 = assemble(a1)

a3 = lhs(F3)
L3 = rhs(F3)
A3 = assemble(a3)

# Define variational problem for solid
F2 = nu_s*inner(grad(disp), grad(vs))*dx + inner(us, vs)*dx
a2 = lhs(F2)
L2 = rhs(F2)
A2 = assemble(a2)

# Use amg preconditioner if available
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"

# Use nonzero guesses - essential for CG with non-symmetric BC
parameters['krylov_solver']['nonzero_initial_guess'] = True

# Create files for storing solution
ufile = File("results/velocity.pvd")
pfile = File("results/pressure.pvd")
ffile = File("results/force.pvd")
dfile = File("results/disp.pvd")

# Time-stepping
t = dt
while t < T + DOLFIN_EPS:
    # step 1. calculate velocity and pressure
    b1 = assemble(L1)
    b3 = assemble(L3)

    [bc.apply(A1, b1) for bc in bcns]
    [bc.apply(A3, b3) for bc in bcsav]
    solve(A1, w1.vector(), b1)
    solve(A3, w2.vector(), b3)
    SS.S = calculate_s(w1,w2,un)
    w_ = project(w1+SS*w2,W)
    (u_, p_) = w_.split(True)
    # step 2. interpolate velocity from fluid to solid
    IB.fluid_to_solid(u_._cpp_object, velocity._cpp_object)
    # step 3. calculate disp for solid and update current gauss points and dof points
    disp.vector()[:] = velocity.vector()[:]*dt + disp.vector()[:]
    IB.evaluate_current_points(disp._cpp_object)
    # step 4. calculate body force.
    b2 = assemble(L2)
    solve(A2, force.vector(), b2)
    # step 5. interpolate force from solid to fluid
    IB.solid_to_fluid(f._cpp_object, force._cpp_object)
    # step 6. update variables and save to file.
    un.assign(u_)
    pn.assign(p_)
    ufile << un
    pfile << pn
    ffile << f
    # step 7. move the mesh.
    dfile << disp
    t += dt
    print(t)

