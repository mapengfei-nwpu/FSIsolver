from StokesSolver import myStokesSolver
from fenics import *
from mshr import *
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
    f = Constant((0,0))

    # output velocity
    ufile = File('NSsolver/u.pvd')
    
    for n in range(100):
        u1, p1 = myStokesSolver(u0, p0, f, bcu, bcp, dt = dt, nu = mu)
        u0.assign(u1)
        p0.assign(p1)
        ufile << u0
        print("step : ",n)

if __name__ == '__main__':
    run_solver()