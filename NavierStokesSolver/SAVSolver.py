from .IPCSSolver import IPCSSolver
from ..solutions import solutions
from fenics import *
import numpy as np
# Set parameter values
n = 32
T = 1.0
num_steps = n*n*2*2
dt = T / num_steps
nu = 0.01
mesh = RectangleMesh(Point(0, 0), Point(1, 1), n, n)

solution = solutions[1]
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

f_exact = Expression((solution["fx"], solution["fy"]), degree=2, t=0)
p_exact = Expression(solution["p"], degree=2, t=0)
u_exact = Expression((solution["ux"], solution["uy"]), degree=2, t=0)
bcu = [DirichletBC(V, u_exact, "on_boundary")]
bcp = [DirichletBC(Q, p_exact, "on_boundary")]
bcu_sav = [DirichletBC(V, Constant((0,0)), "on_boundary")]
bcp_sav = [DirichletBC(V, Constant(0), "on_boundary")]

u0 = Function(V)
p0 = Function(Q)
u0.interpolate(u_exact)
p0.interpolate(p_exact)


ufile = File('test/u.pvd')
k=Constant(dt)
SS = Expression("S", degree=1, S=1.0)
navier_stokes_solver = IPCSSolver(u0, p0, f_exact-grad(u0)*u0, dt = dt, nu = nu)
navier_stokes_solver = IPCSSolver(u0, p0, u0/dt,               dt = dt, nu = nu)

final_step = int(0.1/dt)
for n in range(final_step):
    f_exact.t = n*dt
    u_exact.t = n*dt
    p_exact.t = n*dt
    u1, p1 = navier_stokes_solver.solve(u0, p0, bcu, bcp)
    u2, p2 = navier_stokes_solver.solve(u0, p0, bcu_sav, bcp_sav)
    u_ = project(u1+SS*u2, V)
    u_ = project(p1+SS*p2, V)
    u0.assign(u_)
    p0.assign(p_)
    ufile << u0


print("||u||_2: ", np.sqrt(assemble(inner((u0-u_exact), (u0-u_exact))*dx)))
print("||p||_2: ", np.sqrt(assemble((p1-p_exact)*(p1-p_exact)*dx)))