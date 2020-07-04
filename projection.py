"""Chorin's splitting method."""
from dolfin import *
import numpy as np
from solutions import solutions

# Set dolfin parameters
prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
parameters['krylov_solver']['nonzero_initial_guess'] = True
parameters["std_out_all_processes"] = False


def PrejectionSolve(n, solution):
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), n, n)
    f = Expression((solution["fx"], solution["fy"]), degree=2, t=0)
    p_exact = Expression(solution["p"], degree=2, t=0)
    u_exact = Expression((solution["ux"], solution["uy"]), degree=2, t=0)

    # Define function spaces (P2-P1)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Set parameter values
    dt = 1e-5
    T = 0.1
    nu = 0.01

    # Define boundary conditions
    bcu = [DirichletBC(V, u_exact, "on_boundary")]
    bcp = [DirichletBC(
        Q, p_exact, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")]

    # Create functions
    u0 = Function(V)
    u1 = Function(V)
    p1 = Function(Q)

    # Define coefficients
    k = Constant(dt)

    # Tentative velocity step
    F1 = (1/k)*inner(u - u0, v)*dx + inner(grad(u0)*u0, v)*dx + \
        nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Pressure update
    a2 = inner(grad(p), grad(q))*dx
    L2 = -(1/k)*div(u1)*q*dx

    # Velocity update
    a3 = inner(u, v)*dx
    L3 = inner(u1, v)*dx - k*inner(grad(p1), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Time-stepping
    t = dt
    u0.interpolate(u_exact)
    while t < T + DOLFIN_EPS:

        # Update boundary condition
        p_exact.t = t
        u_exact.t = t
        f.t = t

        # Compute tentative velocity step
        b1 = assemble(L1)
        [bc.apply(A1, b1) for bc in bcu]
        solve(A1, u1.vector(), b1, "bicgstab", "default")

        # Pressure correction
        b2 = assemble(L2)
        [bc.apply(A2, b2) for bc in bcp]
        [bc.apply(p1.vector()) for bc in bcp]
        solve(A2, p1.vector(), b2, "bicgstab", prec)

        # Velocity correction
        b3 = assemble(L3)
        [bc.apply(A3, b3) for bc in bcu]
        solve(A3, u1.vector(), b3, "bicgstab", "default")

        # Move to next time step
        u0.assign(u1)
        t += dt

    # Print errors
    print("||u||_2: ", np.sqrt(assemble(inner((u0-u_exact), (u0-u_exact))*dx)))
    print("||p||_2: ", np.sqrt(assemble((p1-p_exact)*(p1-p_exact)*dx)))



for i in range (4):
    for j in range(4):
        PrejectionSolve(pow(2,1+j), solutions[i])