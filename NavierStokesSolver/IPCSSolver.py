from fenics import *
from mshr import *
import numpy as np

class IPCSSolver:
    def __init__(self, u0, p0, f, dt = 0.01, nu = 0.01):        
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
        self.u_  = Function(V)
        self.p_n = Function(Q)
        self.p_  = Function(Q)

        # Define expressions used in variational forms
        U  = 0.5*(self.u_n + u)
        n  = FacetNormal(mesh)
        k  = Constant(dt)
        rho = Constant(1)

        # Define symmetric gradient
        def epsilon(u):
            return sym(nabla_grad(u))

        # Define stress tensor
        def sigma(u, p):
            return 2*nu*epsilon(u) - p*Identity(len(u))

        # Define variational problem for step 1
        F1 = rho*dot((u - self.u_n) / k, v)*dx \
            + rho*dot(dot(self.u_n, nabla_grad(self.u_n)), v)*dx \
            + inner(sigma(U, self.p_n), epsilon(v))*dx \
            + dot(self.p_n*n, v)*ds - dot(nu*nabla_grad(U)*n, v)*ds \
            - dot(f, v)*dx
        a1 = lhs(F1)
        self.L1 = rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        self.L2 = dot(nabla_grad(self.p_n), nabla_grad(q))*dx - (1/k)*div(self.u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        self.L3 = dot(self.u_, v)*dx - k*dot(nabla_grad(self.p_ - self.p_n), v)*dx

        # Assemble matrices
        self.A1 = assemble(a1)
        self.A2 = assemble(a2)
        self.A3 = assemble(a3)

    def solve(self, u0, p0, bcu, bcp):
        self.u_n.assign(u0)
        self.p_n.assign(p0)

        # Step 1: Tentative velocity step
        b1 = assemble(self.L1)
        [bc.apply(self.A1, b1) for bc in bcu]
        solve(self.A1, self.u_.vector(), b1, 'bicgstab', 'hypre_amg')

        # Step 2: Pressure correction step
        b2 = assemble(self.L2)
        [bc.apply(self.A2, b2) for bc in bcp]
        solve(self.A2, self.p_.vector(), b2, 'bicgstab', 'hypre_amg')

        # Step 3: Velocity correction step
        b3 = assemble(self.L3)
        solve(self.A3, self.u_.vector(), b3, 'cg', 'sor')
        return self.u_, self.p_
def run_solver():
    "Run solver to compute and post-process solution"
    T = 5.0
    num_steps = 5000
    dt = T / num_steps
    mu = 0.01
    # Create mesh
    channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
    cylinder = Circle(Point(0.2, 0.2), 0.05)
    domain = channel - cylinder
    mesh = generate_mesh(domain, 32)

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
    f = Constant((0,0))

    # output velocity
    ufile = File('IPCSSolver/u.pvd')
    navier_stokes_solver = IPCSSolver(u0, p0, f, dt = dt, nu = mu)
    
    for n in range(1000):
        u1, p1 = navier_stokes_solver.solve(u0, p0, bcu, bcp)
        u0.assign(u1)
        p0.assign(p1)
        ufile << u0
        print("step : ", n)

if __name__ == '__main__':
    run_solver()