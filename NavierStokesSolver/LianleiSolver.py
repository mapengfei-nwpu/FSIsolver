# 二阶
from fenics import *
from mshr import *
import numpy as np

def s_solve(C0,A0,A1,A2,B0,B1,B2,dt):
    S0 = 1
    SN = 0
    while(abs(SN-S0)>1e-6):
        SN = S0
        ESN   = A0 + A1*SN + A2*SN*SN
        ESND  = A1 + 2*A2*SN
        SESN  = sqrt(ESN)
        R = SESN
        SESND = SESN/(2.0*SESN)

        FSN  = 2.0/dt*(SN*SN*SN*ESN) - 2.0/dt*R*(SN*SN*SESN) + B0*SN + B1*SN*SN + B2*SN*SN*SN
        FSND = 2.0/dt*(3.0*SN*SN*ESN + SN*SN*SN*ESND) - 2.0/dt*R*(2*SN*SESN + SN*SN*SESND) + B0 + 2.0*B1*SN + 3.0*B2*SN*SN
        S0 = SN - FSN/FSND

        print("S = ", S0)

    return S0

class LianleiSolver:
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
        self.f = f
        self.k = Expression("dt", degree=1, dt=dt)
        self.nu = Expression("nu", degree=1, nu=nu)
        self.n = FacetNormal(mesh)

        # TODO : F1 缺少一项边界项
        F1 = -inner(grad(p), grad(q))*dx + inner(self.f + self.u_n/self.k, grad(q)) * dx - 1/self.k*inner(self.n, self.u_n)*q*ds

        F2 = -inner(grad(p), grad(q))*dx - inner(grad(self.u_n)*self.u_n, grad(q))*dx

        F3 = 1/self.k/self.nu*inner(u, v)*dx + inner(grad(u), grad(v)) * dx \
             - 1/self.nu*inner(self.f+self.u_n/self.k-grad(self.p_1), v)*dx

        F4 = 1/self.k/self.nu*inner(u, v)*dx + inner(grad(u), grad(v)) * dx \
             + 1/self.nu*inner(grad(self.u_n)*self.u_n+grad(self.p_2), v)*dx

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

    def solve(self, u0, p0, bcu, bcp, dt=0.01, nu=0.01):

        # Update variables.
        self.u_n.assign(u0)
        self.p_n.assign(p0)
        self.k.dt = dt
        self.nu.nu = nu

        # Step 1: 
        b1 = assemble(self.L1)
        [bc.apply(self.A1, b1) for bc in bcp]
        solve(self.A1, self.p_1.vector(), b1, 'bicgstab', 'default')

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

        C0 = 0.1
        A0 = assemble(0.5*inner(self.u_1, self.u_1)*dx) + C0
        A1 = assemble(inner(self.u_1, self.u_2)*dx)
        A2 = assemble(0.5*inner(self.u_2, self.u_2)*dx) 

        B0 =     nu*assemble(inner(grad(self.u_1), grad(self.u_1))*dx) - assemble(inner(self.f,self.u_1)*dx) # - assemble(*ds)
        B1 = 2.0*nu*assemble(inner(grad(self.u_1), grad(self.u_2))*dx) - assemble(inner(self.f,self.u_2)*dx)
        B2 =     nu*assemble(inner(grad(self.u_2), grad(self.u_2))*dx)

        S = 1.0 # s_solve(C0,A0,A1,A2,B0,B1,B2,dt)

        self.u_.vector()[:] = self.u_1.vector()[:] + S*self.u_2.vector()[:]
        self.p_.vector()[:] = self.p_1.vector()[:] + S*self.p_2.vector()[:]
        
        return self.u_, self.p_


def run_solver():
    "Run solver to compute and post-process solution"
    T = 5.0
    num_steps = 5000
    dt = T / num_steps
    nu = 0.001
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
    ufile = File('LianleiSolver/u.pvd')
    navier_stokes_solver = LianleiSolver(u0, p0, f, dt=dt, nu=nu)

    for n in range(5000):
        u1, p1 = navier_stokes_solver.solve(u0, p0, bcu, bcp)
        u0.assign(u1)
        p0.assign(p1)
        ufile << u0
        print("step : ", n)


if __name__ == '__main__':
    run_solver()
