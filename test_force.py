from NavierStokesSolver import *
from DeltaInterpolation3D import *
from dolfin import *
from mshr import *

# solid_mesh = generate_mesh(Sphere(Point(0.6, 0.5, 0.5), 0.2),8)
solid_mesh = generate_mesh(Circle(Point(0.6, 0.5), 0.2), 10)
Vs = VectorFunctionSpace(solid_mesh, "Lagrange", 1)
VsT = TensorFunctionSpace(solid_mesh, "Lagrange", 1)

# Define trial and test functions for solid
us = TrialFunction(Vs)
vs = TestFunction(Vs)
disp = Function(Vs)

# disp.interpolate(Expression(("x[0]","x[1]","x[2]"),degree=1))
disp.interpolate(Expression(("x[0]","x[1]"),degree=1))

F = project(grad(disp), VsT)
File("F.pvd") << F


force = Function(Vs)

# Define variational problem for solid
F2 = 0.1*inner(grad(disp)-inv(grad(disp)).T, grad(vs))*dx + inner(us, vs)*dx
a2 = lhs(F2)
L2 = rhs(F2)
A2 = assemble(a2)
b2 = assemble(L2)
solve(A2, force.vector(), b2)
File("force0.pvd") << force

# Define variational problem for solid-inv(grad(disp)).T
F2 = 0.1*inner(grad(disp), grad(vs))*dx + inner(us, vs)*dx
a2 = lhs(F2)
L2 = rhs(F2)
A2 = assemble(a2)
b2 = assemble(L2)
solve(A2, force.vector(), b2)
File("force1.pvd") << force
