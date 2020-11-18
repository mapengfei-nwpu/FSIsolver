from fenics import *
def first_PK_stress(u):                       # input the displacement

    C = Constant(10000.0)                      # C = 10kPa
    bf = Constant(1.0)
    bt = Constant(1.0)
    bfs = Constant(1.0)

    I = Identity(len(u))
    F = variable(I + grad(u))                         # nabla_grad is used someplaces. I think grad is correct.
    E = 0.5*(F.T*F - I)

    e1 = as_vector([ 1.0, 0.0, 0.0 ])
    e2 = as_vector([ 0.0, 1.0, 0.0 ])
    e3 = as_vector([ 0.0, 0.0, 1.0 ])

    E11, E12, E13 = inner(E*e1, e1), inner(E*e1, e2), inner(E*e1, e3)
    E21, E22, E23 = inner(E*e2, e1), inner(E*e2, e2), inner(E*e2, e3)
    E31, E32, E33 = inner(E*e3, e1), inner(E*e3, e2), inner(E*e3, e3)

    Q = bf*E11**2 + bt*(E22**2 + E33**2 + E23**2 + E32**2) \
      + bfs*(E12**2 + E21**2 + E13**2 + E31**2)

    Wpassive = C/2.0 * (exp(Q) - 1)

    return diff(Wpassive, F)       # Calculate the first PK stress tensor

    # C = variable(F.T*F)          # calculate the right Cauchy-Green tensor
    # S=2*diff(Wpassive,C)         # calculate the second PK stress tensor
    # return F*S                   # Calculate the first PK stress tensor

#TODO: Active stress tensor.
#TODO: Multiplier on base at the z direction.
#TODO: Pressure on the inner wall.
#TODO: Unit conversion: mesh size, viscous coefficient, pressure.

if __name__ == "__main__":
    from LeftVentricleMesh import mesh 
    # mesh = UnitCubeMesh(10,10,10)
    V    = VectorFunctionSpace(mesh,"P",2)
    u = TrialFunction(V)
    v = TestFunction(V)
    disp = interpolate(Expression(("x[0]","x[1]","x[2]"),degree=2), V)
    F = inner(first_PK_stress(disp), grad(v))*dx + inner(u, v)*dx
    a = lhs(F)
    L = rhs(F)
    A = assemble(a)
    b = assemble(L)
    n = Function(V)
    solve(A, n.vector(), b, 'cg', 'sor' )
    File("n.pvd") << n