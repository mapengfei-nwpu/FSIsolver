from NavierStokesSolver import IPCSSolver
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

# Set variables
n_mesh = 32              # define the mesh size.
epsilon = 1/n_mesh       # the width of the margin of the domain.
dt = 0.125/n_mesh
T = 10
n_step = int(T/dt)
nu = 0.01
mu_s = 0.1
delta = 0.1


class Heaviside(UserExpression):
    def eval(self, values, x):
        level_set = 0.2 - \
            np.sqrt((x[0]-0.6)*(x[0]-0.6) + (x[1]-0.5)*(x[1]-0.5) + (x[2]-0.5)*(x[2]-0.5))
        if level_set > epsilon:
            values[0] = 1
        elif level_set < -epsilon:
            values[0] = 0
        else:
            values[0] = 0.5*(1.0 + np.sin(np.pi*level_set/2.0/epsilon))


heaviside = Heaviside()

# Define mesh
mesh = UnitCubeMesh(n_mesh, n_mesh, n_mesh)

# Define function spaces V_s, V_v, V_t
V_s = FunctionSpace(mesh, "Lagrange", 1)
V_v = VectorFunctionSpace(mesh, "Lagrange", 2)
V_t = TensorFunctionSpace(mesh, "Lagrange", 2)

# Define boundary
noslip = DirichletBC(
    V_v, (0, 0, 0), "near(x[0],1) || near(x[0],0) || near(x[1],0)")
upflow = DirichletBC(V_v, (1, 0, 0), "near(x[1],1)")
pinpoint = DirichletBC(V_s, 0, "near(x[0],0) && near(x[1],0) && near(x[2],0)", "pointwise")
bcu = [noslip, upflow]                                     # velocity condition
bcp = [pinpoint]                                           # pressure condition
bch = [DirichletBC(V_s, 0,                "on_boundary")]  # heaviside function
bcf = [DirichletBC(V_v, (0, 0, 0),           "on_boundary")]  # force function
bcg = [DirichletBC(V_t, ((0, 0, 0), (0, 0, 0), (0, 0, 0)), "on_boundary")]  # deformation gradient

# Define variables
v_t = TestFunction(V_t)
v_v = TestFunction(V_v)
u_t = TrialFunction(V_t)
u_v = TrialFunction(V_v)

h = interpolate(heaviside, V_s)
g = project(as_matrix(((h, 0, 0), (0, h, 0), (0, 0, h))), V_t)
f = Function(V_v)

u = Function(V_v)
p = Function(V_s)
k = Constant(dt)

# Define evolving function of deformation gradient.
c = 0.5
theta = 0.5
vnorm = sqrt(dot(u, u)) + 0.001
stable_v_t = v_t + c*mesh.hmin()*dot(u, nabla_grad(v_t))/vnorm
F4 = (1/dt)*inner(u_t-g, stable_v_t)*dx + theta*(inner(dot(u,nabla_grad(u_t)),stable_v_t) - inner(grad(u)*u_t,stable_v_t))*dx + (1-theta)*(inner(dot(u,nabla_grad(g)),stable_v_t)  - inner(grad(u)*g,stable_v_t))*dx
a4 = lhs(F4)
L4 = rhs(F4)

F0 = mu_s*inner((g*g.T-Identity(2)), grad(v_v))*dx + inner(u_v, v_v)*dx
a0 = lhs(F0)
A0 = assemble(a0)
L0 = rhs(F0)

# Directory name
directory = "results3D/meshsize_" + str(n_mesh) +"/mu_s_" + str(mu_s)
# Create files for storing solution
ufile = File(directory + "/velocity.pvd")
pfile = File(directory + "/pressure.pvd")
gfile = File(directory + "/g.pvd")
hfile = File(directory + "/h.pvd")
navier_stokes_solver = IPCSSolver(u, p, f, dt=dt, nu=nu)

for n in range(n_step):
    # Deformation gradient evolving function
    A4 = assemble(a4)
    b4 = assemble(L4)
    [bc.apply(A4, b4) for bc in bcg]
    solve(A4, g.vector(), b4, "gmres", "default")
    # force function
    A0 = assemble(a0)
    b0 = assemble(L0)
    # [bc.apply(A0, b0) for bc in bcf]
    solve(A0, f.vector(), b0, "gmres", "default")

    u1, p1 = navier_stokes_solver.solve(u, p, bcu, bcp)
    u.assign(u1)
    p.assign(p1)

    if n % 100 == 0 :
        ufile << u
        pfile << p
        gfile << g
        hfile << h
        print("step : ", n)
