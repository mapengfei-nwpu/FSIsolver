from fenics import *
from mshr import *
import numpy as np

class NavierStokesSolver:
    def __init__(self, u0, p0, bcu, bcp, dt = 0.01, nu = 0.01):
        # Define function spaces (P2-P1)
        V = u0.function_space()
        Q = p0.function_space()

        # Define trial and test functions
        u = TrialFunction(V)
        p = TrialFunction(Q)
        v = TestFunction(V)
        q = TestFunction(Q)    

        # Create functions
        self.u0 = Function(V)
        self.p0 = Function(Q)
        self.u1 = Function(V)
        self.p1 = Function(Q)
        self.f = Function(V)

        # Define coefficients
        k = Constant(dt)

        # Tentative velocity step
        F1 = (1/k)*inner(u - self.u0, v)*dx + inner(grad(self.u0)*self.u0, v)*dx + \
            nu*inner(grad(u), grad(v))*dx - inner(self.f, v)*dx
        a1 = lhs(F1)
        self.L1 = rhs(F1)

        # Pressure update
        a2 = inner(grad(p), grad(q))*dx
        self.L2 = -(1/k)*div(self.u1)*q*dx

        # Velocity update
        a3 = inner(u, v)*dx
        self.L3 = inner(self.u1, v)*dx - k*inner(grad(self.p1), v)*dx

        # Assemble matrices
        self.A1 = assemble(a1)
        self.A2 = assemble(a2)
        self.A3 = assemble(a3)

    def solve(self, u0, p0, f, bcu, bcp):
        self.u0.assign(u0)
        self.p0.assign(p0)
        self.f.assign(f)

        # Compute tentative velocity step
        b1 = assemble(self.L1)
        [bc.apply(self.A1, b1) for bc in bcu]
        solve(self.A1, self.u1.vector(), b1, "bicgstab", "default")

        # Pressure correction
        b2 = assemble(self.L2)
        [bc.apply(self.A2, b2) for bc in bcp]
        solve(self.A2, self.p1.vector(), b2, "bicgstab", "amg")

        # Velocity correction
        b3 = assemble(self.L3)
        [bc.apply(self.A3, b3) for bc in bcu]
        solve(self.A3, self.u1.vector(), b3, "bicgstab", "default")

        return self.u1, self.p1

def run_solver():
    "Run solver to compute and post-process solution"
    T = 5.0
    num_steps = 5000
    dt = T / num_steps
    mu = 0.001
    # Create mesh
    channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
    cylinder = Circle(Point(0.2, 0.2), 0.05)
    domain = channel - cylinder
    mesh = generate_mesh(domain, 64)

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)

    # Define boundaries
    inflow =    'near(x[0], 0)'
    outflow =   'near(x[0], 2.2)'
    walls =     'near(x[1], 0) || near(x[1], 0.41)'
    cylinder =  'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

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
    f = Function(V)

    # output velocity
    ufile = File('NavierStokesSolver/u.pvd')
    navier_stokes_solver = NavierStokesSolver(u0, p0, bcu, bcp, dt = dt, nu = mu)
    
    for n in range(100):
        u1, p1 = navier_stokes_solver.solve(u0, p0, f, bcu, bcp)
        u0.assign(u1)
        p0.assign(p1)
        ufile << u0
        print("step : ", n)

if __name__ == '__main__':
    run_solver()