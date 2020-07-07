from fenics import *
from mshr import *
import numpy as np


class IPCSSolver:
    def __init__(self, u0, p0, f, dt=0.01, nu=0.01):
        # Define function spaces (P2-P1)
        V = u0.function_space()
        Q = p0.function_space()
        mesh = V.mesh()

        # Define trial and test functions
        u = TrialFunction(V)
        p = TrialFunction(Q)
        v = TestFunction(V)
        q = TestFunction(Q)

        # Define functions for solutions at previous and current time steps
        self.u_n = Function(V)
        self.u_ = Function(V)
        self.p_n = Function(Q)
        self.p_ = Function(Q)
        self.u_1 = Function(V)
        self.u_2 = Function(V)
        self.p_1 = Function(Q)
        self.p_2 = Function(Q)

        # Define SAV boundary conditions
        self.bcu_sav = [DirichletBC(V, Constant((0, 0)), "on_boundary")]
        self.bcp_sav = []

        # Define expressions used in variational forms
        k = Constant(dt)
        n = FacetNormal(mesh)

        F1 = -inner(grad(p), grad(q))*dx + inner(f + self.u_n/k, grad(q)) * \
            dx - 1/k*inner(n, self.u_n)*q*ds

        F2 = -inner(grad(p), grad(q))*dx - inner(grad(self.u_n)*self.u_n, grad(q))*dx

        F3 = 1/k/nu*inner(u, v)*dx + inner(grad(u), grad(v)) * dx \
             - 1/nu*inner(f+self.u_n/k-grad(self.p_1), v)*dx

        F4 = 1/k/nu*inner(u, v)*dx + inner(grad(u), grad(v)) * dx \
             + 1/nu*inner(grad(self.u_n)*self.u_n+grad(self.p_2), v)*dx

        a1 = lhs(F1)
        a2 = lhs(F2)
        a3 = lhs(F3)
        a4 = lhs(F4)

        self.L1 = rhs(F1)
        self.L2 = rhs(F2)
        self.L3 = rhs(F3)
        self.L4 = rhs(F4)

        self.A1 = assemble(a1)
        self.A2 = assemble(a2)
        self.A3 = assemble(a3)
        self.A4 = assemble(a4)

    def solve(self, u0, p0, bcu, bcp):
        self.u_n.assign(u0)
        self.p_n.assign(p0)

        # Step 1: 
        b1 = assemble(self.L1)
        [bc.apply(self.A1, b1) for bc in bcp]
        solve(self.A1, self.p_1.vector(), b1, 'cg', 'hypre_amg')

        # Step 2: 
        b2 = assemble(self.L2)
        [bc.apply(self.A2, b2) for bc in self.bcp_sav]
        solve(self.A2, self.p_2.vector(), b2, 'bicgstab', 'default')

        # Step 3: 
        b3 = assemble(self.L3)
        [bc.apply(self.A3, b3) for bc in bcu]
        solve(self.A3, self.u_1.vector(), b3, 'bicgstab', 'default')

        # Step 4: 
        b4 = assemble(self.L4)
        [bc.apply(self.A4, b4) for bc in self.bcu_sav]
        solve(self.A4, self.u_2.vector(), b4, 'bicgstab', 'default')

        self.u_.vector()[:] = self.u_1.vector()[:] + self.u_2.vector()[:]
        self.p_.vector()[:] = self.p_1.vector()[:] + self.p_2.vector()[:]

        return self.u_, self.p_


def run_solver():
    "Run solver to compute and post-process solution"
    T = 5.0
    num_steps = 5000
    dt = T / num_steps
    nu = 0.01
    # Create mesh
    channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
    cylinder = Circle(Point(0.2, 0.2), 0.05)
    domain = channel - cylinder
    mesh = generate_mesh(domain, 64)

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)

    # Define boundaries
    inflow = 'near(x[0], 0)'
    outflow = 'near(x[0], 2.2)'
    walls = 'near(x[1], 0) || near(x[1], 0.41)'
    cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

    # Define inflow profile
    inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

    # Define boundary conditions
    bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
    bcp_outflow = DirichletBC(Q, Constant(0), outflow)
    bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
    bcp = [bcp_outflow]

    u0 = Function(V)
    p0 = Function(Q)
    f = Constant((0, 0))

    # output velocity
    ufile = File('IPCSSolver/u.pvd')
    navier_stokes_solver = IPCSSolver(u0, p0, f, dt=dt, nu=nu)

    for n in range(1000):
        u1, p1 = navier_stokes_solver.solve(u0, p0, bcu, bcp)
        u0.assign(u1)
        p0.assign(p1)
        ufile << u0
        print("step : ", n)


if __name__ == '__main__':
    run_solver()
