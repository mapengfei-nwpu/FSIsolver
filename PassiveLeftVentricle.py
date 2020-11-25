from fenics import *



def first_PK_stress(u):                       # input the displacement

    C = Constant(10.0)                      # C = 10kPa
    bf = Constant(1.0)
    bt = Constant(1.0)
    bfs = Constant(1.0)

    I = Identity(len(u))
    F = variable(grad(u))                         # nabla_grad is used someplaces. I think grad is correct.
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

    return diff(Wpassive, F)

if __name__ == "__main__":
    from LeftVentricleMesh import mesh
    V    = VectorFunctionSpace(mesh,"P",2)
    u = TrialFunction(V)
    v = TestFunction(V)
    disp = interpolate(Expression(("x[0]","x[1]","x[2]"),degree=3), V)
    F = inner(first_PK_stress(disp), grad(v))*dx + inner(u, v)*dx
    a = lhs(F)
    L = rhs(F)
    A = assemble(a)
    b = assemble(L)
    n = Function(V)
    solve(A, n.vector(), b, "cg", "sor")
    File("n_1.pvd") << n