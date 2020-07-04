from dolfin import *
from mshr import *
import numpy as np

class TaylorHoodSolver:
    def __init__(self, u0, p0, dt = 0.01, nu = 0.01):
        # Reconstruct element space
        mesh = u0.function_space().mesh()
        element1 = u0.function_space()._ufl_element
        element2 = p0.function_space()._ufl_element
        TH = element1*element2
        W = FunctionSpace(mesh, TH)

        # Define variables
        (u, p) = TrialFunctions(W)
        (v, q) = TestFunctions(W)
        k = Constant(dt)
        self.w_ = Function(W)
        self.un, self.pn = Function(W).split(True)
        self.f,_ = Function(W).split(True)
        
        # Define variational problem
        F = inner((u-self.un)/k, v)*dx + inner(grad(self.un)*self.un, v)*dx + nu * inner(grad(u), grad(v))*dx - div(v)*p * dx + q*div(u)*dx - inner(self.f, v)*dx
        a = lhs(F)
        self.A = assemble(a)
        self.L = rhs(F)

    def solve(self, u0, p0, f, bcu, bcp):
        self.un.assign(u0)
        self.pn.assign(p0)
        self.f.assign(f)

        # Compute tentative velocity step
        b = assemble(self.L)
        [bc.apply(self.A, b) for bc in bcu]
        [bc.apply(self.A, b) for bc in bcp]
        solve(self.A, self.w_.vector(), b)

        return self.w_.split(True)

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
    P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = P2 * P1
    W = FunctionSpace(mesh, TH)

    # Define boundaries
    inflow =    'near(x[0], 0)'
    outflow =   'near(x[0], 2.2)'
    walls =     'near(x[1], 0) || near(x[1], 0.41)'
    cylinder =  'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

    # Define inflow profile
    inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

    # Define boundary conditions
    bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2), inflow)
    bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls)
    bcu_cylinder = DirichletBC(W.sub(0), Constant((0, 0)), cylinder)
    bcp_outflow = DirichletBC(W.sub(1), Constant(0), outflow)
    bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
    bcp = [bcp_outflow]
    
    u0,p0 = Function(W).split(True)
    f,_   = Function(W).split(True)

    # output velocity
    ufile = File('TaylorHoodSolver/u.pvd')
    navier_stokes_solver = TaylorHoodSolver(u0, p0, dt = dt, nu = nu)
    
    for n in range(100):
        u1, p1 = navier_stokes_solver.solve(u0, p0, f, bcu, bcp)
        u0.assign(u1)
        p0.assign(p1)
        ufile << u0
        print("step : ", n)

if __name__ == '__main__':
    run_solver()