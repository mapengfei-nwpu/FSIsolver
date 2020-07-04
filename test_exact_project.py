from NavierStokesSolver import *
from solutions import solutions
from fenics import *
import numpy as np
# Set parameter values
n = 32
T = 1.0
num_steps = n*n*2*2
dt = T / num_steps
mu = 0.01
mesh = RectangleMesh(Point(0, 0), Point(1, 1), n, n)

solution = solutions[1]
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

f_exact = Expression((solution["fx"], solution["fy"]), degree=2, t=0)
p_exact = Expression(solution["p"], degree=2, t=0)
u_exact = Expression((solution["ux"], solution["uy"]), degree=2, t=0)
bcu = [DirichletBC(V, u_exact, "on_boundary")]
bcp = [DirichletBC(Q, p_exact, "on_boundary")]

u0 = Function(V)
p0 = Function(Q)
f = Function(V)
u0.interpolate(u_exact)
p0.interpolate(p_exact)
f.interpolate(f_exact)

ufile = File('test/u.pvd')
navier_stokes_solver = NavierStokesSolver(u0, p0, bcu, bcp, dt = dt, nu = mu)

final_step = int(0.1/dt)
for n in range(final_step):
    f_exact.t = n*dt
    u_exact.t = n*dt
    p_exact.t = n*dt
    f.interpolate(f_exact)
    u1, p1 = navier_stokes_solver.solve(u0, p0, f, bcu, bcp)
    u0.assign(u1)
    p0.assign(p1)
    ufile << u0


print("||u||_2: ", np.sqrt(assemble(inner((u0-u_exact), (u0-u_exact))*dx)))
print("||p||_2: ", np.sqrt(assemble((p1-p_exact)*(p1-p_exact)*dx)))