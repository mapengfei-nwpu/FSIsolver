from NavierStokesSolver import *
from solutions import solutions
from fenics import *
import numpy as np
# Set parameter values
delta = 0.1
n = 4
T = 1.0
num_steps = n*n*2*2
dt = T / num_steps
nu = 0.01
mesh = RectangleMesh(Point(0, 0), Point(1, 1), n, n)

solution = solutions[1]
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

f_exact = Expression((solution["fx"], solution["fy"]), degree=2, t=0)
p_exact = Expression(solution["p"], degree=2, t=0)
u_exact = Expression((solution["ux"], solution["uy"]), degree=2, t=0)
bcu = [DirichletBC(W.sub(0), u_exact, "on_boundary")]
bcp = [DirichletBC(W.sub(1), p_exact, "on_boundary")]
bcu_sav = [DirichletBC(W.sub(0), Constant((0,0)), "on_boundary")]
bcp_sav = [DirichletBC(W.sub(1), Constant(0),     "on_boundary")]

u0,p0 = Function(W).split(True)
u0.interpolate(u_exact)
p0.interpolate(p_exact)

ufile = File('test/u.pvd')
k=Constant(dt)
SS = Expression("S", degree=1, S=1.0)
f1 = f_exact+grad(u0)*u0
f2 = - u0/k
navier_stokes_solver_1 = TaylorHoodSolver(u0, p0, f1, dt = dt, nu = nu)
navier_stokes_solver_2 = TaylorHoodSolver(u0, p0, f2, dt = dt, nu = nu)

final_step = int(0.1/dt)

def calculate_s(u1,u2,un,f):
    temp = 0.5*assemble(inner(un, un)*dx) + delta
    r = sqrt(temp)
    a2 = nu*assemble(inner(grad(u2), grad(u2))*dx) + 2/dt*temp
    a1 = 2*nu*assemble(inner(grad(u2), grad(u1))*dx) - 2 * \
        r/dt*sqrt(temp)-assemble(inner(f, u2)*dx)
    a0 = nu*assemble(inner(grad(u1), grad(u1))*dx) - assemble(inner(f, u1)*dx)
    s1 = (-a1+sqrt(a1*a1-4*a2*a0))/2/a2
    s2 = (-a1-sqrt(a1*a1-4*a2*a0))/2/a2
    if max(s1, s2) > 0 and min(s1, s2) < 0:
        return max(s1, s2)
    if abs(s1-1) > abs(s2-1):
        return s2
    else:
        return s1

for n in range(final_step):
    f_exact.t = n*dt
    u_exact.t = n*dt
    p_exact.t = n*dt
    u1, p1 = navier_stokes_solver_1.solve(u0, p0, bcu,     bcp)
    u2, p2 = navier_stokes_solver_2.solve(u0, p0, bcu_sav, bcp_sav)
    SS.S = calculate_s(u1,u2,u0,f_exact)
    u_ = project(u1 + SS*u2, V)
    p_ = project(p1 + SS*p2, Q)
    u0.assign(u_)
    p0.assign(p_)
    ufile << u0


print("||u||_2: ", np.sqrt(assemble(inner((u0-u_exact), (u0-u_exact))*dx)))
print("||p||_2: ", np.sqrt(assemble((p0-p_exact)*(p0-p_exact)*dx)))