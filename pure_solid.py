from fenics import *
from LeftVentricleMesh     import mesh          as solid_mesh
from LeftVentricleMesh     import mesh_function as solid_boundary
from PassiveLeftVentricle  import first_PK_stress
from LeftVentricleBoundaryCondition import apply_boundary_conditions

# the markers of the boundary
marker_base = 1
marker_wall = 2

# Define function space for left ventricle
Vs = VectorFunctionSpace(solid_mesh, "Lagrange", 2)

# Define trial and test functions for solid
us = TrialFunction(Vs)
vs = TestFunction(Vs)

# Define the variables
force    = Function(Vs)
disp     = Function(Vs)
velocity = Function(Vs)

# the displacement at the starter
disp.interpolate(Expression(("x[0]", "x[1]", "x[2]"), degree=2))

# Define the variational formulation for the solid mechanics
F = inner(first_PK_stress(disp), grad(vs))*dx + inner(us, vs)*dx
a = lhs(F)
L = rhs(F)
A = assemble(a)

# Define the output files
file_velocity = File("solid/velocity.pvd")
file_force = File("solid/force.pvd")
file_disp = File("solid/disp.pvd")


# solver start.
T = 10
dt = 1e-5
t = dt
while t < T + DOLFIN_EPS:
    # calculate the feedback force at the current displacement
    b = assemble(L)
    solve(A, force.vector(), b, 'cg', 'sor' )
    # apply the boundary conditions at the force
    apply_boundary_conditions(disp,force,solid_boundary,marker_base,marker_wall)
    # update the velocity and position
    velocity.vector()[:] = force.vector()[:]*dt + velocity.vector()[:]
    disp.vector()[:] = velocity.vector()[:]*dt + disp.vector()[:]
    # output
    file_disp << disp
    file_force << force
    file_velocity << velocity
    # update the time
    t += dt
    print(t)