from solutions import solutions
from dolfin import *
import numpy as np

prec = "amg" if has_krylov_solver_preconditioner("amg") else "default"
parameters['krylov_solver']['nonzero_initial_guess'] = True

def SAVPrejectionSolve(n, solution):

    # Set exact solutions
    f = Expression((solution["fx"], solution["fy"]), degree=2, t=0)
    p_exact = Expression(solution["p"], degree=2, t=0)
    u_exact = Expression((solution["ux"], solution["uy"]), degree=2, t=0)

    # Set constant values
    dt = 1e-5
    T  = 0.1
    nu = 0.01
    delta = 0.1
    k = Constant(dt)

    # Define function spaces (P2-P1)
    mesh = RectangleMesh(Point(0, 0), Point(1, 1), n, n)
    V = VectorFunctionSpace(mesh, "Lagrange", 2)
    Q = FunctionSpace(mesh, "Lagrange", 1)

    # Define trial and test functions
    u = TrialFunction(V)
    p = TrialFunction(Q)
    v = TestFunction(V)
    q = TestFunction(Q)

    # Define boundary conditions
    bcu1 = [DirichletBC(V, u_exact, "on_boundary")]
    bcu2 = [DirichletBC(V, (0, 0), "on_boundary")]
    bcp1 = [DirichletBC(Q, p_exact, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")]
    bcp2 = [DirichletBC(Q, 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")]
    
    class SAVSolver1:
        def __init__(self):
            # Create functions
            self.u0 = Function(V)
            self.u1 = Function(V)
            self.p1 = Function(Q)
            F1 = (1/k)*inner(u - self.u0, v)*dx + nu*inner(grad(u), grad(v))*dx - inner(f, v)*dx
            a1 = lhs(F1)
            a2 = inner(grad(p), grad(q))*dx
            a3 = inner(u, v)*dx
            self.L1 = rhs(F1)
            self.L2 = -(1/k)*div(self.u1)*q*dx
            self.L3 = inner(self.u1, v)*dx - k*inner(grad(self.p1), v)*dx
            # Assemble matrices
            self.A1 = assemble(a1)
            self.A2 = assemble(a2)
            self.A3 = assemble(a3)

        def solve(self, u0):
            self.u0.assign(u0)
            # Compute tentative velocity step
            b1 = assemble(self.L1)
            [bc.apply(self.A1, b1) for bc in bcu1]
            solve(self.A1, self.u1.vector(), b1, "bicgstab", "default")
            # Pressure correction
            b2 = assemble(self.L2)
            [bc.apply(self.A2, b2) for bc in bcp1]
            solve(self.A2, self.p1.vector(), b2, "bicgstab", prec)
            # Velocity correction
            b3 = assemble(self.L3)
            [bc.apply(self.A3, b3) for bc in bcu1]
            solve(self.A3, self.u1.vector(), b3, "bicgstab", "default")
            return self.u1, self.p1


    class SAVSolver2:
        def __init__(self):
            # Create functions
            self.u0 = Function(V)
            self.u1 = Function(V)
            self.p1 = Function(Q)
            F1 = (1/k)*inner(u, v)*dx + nu*inner(grad(u), grad(v)) * \
                dx + inner(grad(self.u0)*self.u0, v)*dx
            a1 = lhs(F1)
            a2 = inner(grad(p), grad(q))*dx
            a3 = inner(u, v)*dx
            self.L1 = rhs(F1)
            self.L2 = -(1/k)*div(self.u1)*q*dx
            self.L3 = inner(self.u1, v)*dx - k*inner(grad(self.p1), v)*dx
            # Assemble matrices
            self.A1 = assemble(a1)
            self.A2 = assemble(a2)
            self.A3 = assemble(a3)

        def solve(self, u0):
            self.u0.assign(u0)
            # Compute tentative velocity step
            b1 = assemble(self.L1)
            [bc.apply(self.A1, b1) for bc in bcu2]
            solve(self.A1, self.u1.vector(), b1, "bicgstab", "default")
            # Pressure correction
            b2 = assemble(self.L2)
            [bc.apply(self.A2, b2) for bc in bcp2]
            solve(self.A2, self.p1.vector(), b2, "bicgstab", prec)
            # Velocity correction
            b3 = assemble(self.L3)
            [bc.apply(self.A3, b3) for bc in bcu2]
            solve(self.A3, self.u1.vector(), b3, "bicgstab", "default")
            return self.u1, self.p1


    def S_solve(u0, u1, u2):
        temp = 0.5*assemble(inner(u0, u0)*dx) + delta
        r = sqrt(temp)
        a2 = nu*assemble(inner(grad(u2), grad(u2))*dx) + 0.5*dt*temp
        a1 = 2*nu*assemble(inner(grad(u2), grad(u1))*dx) - 2*r/dt*sqrt(temp)
        a0 = nu*assemble(inner(grad(u1), grad(u1))*dx)
        s1 = (-a1+sqrt(a1*a1-4*a2*a0))/2/a2
        s2 = (-a1-sqrt(a1*a1-4*a2*a0))/2/a2
        if max(s1, s2) > 0 and min(s1, s2) < 0:
            return max(s1, s2)
        if abs(s1-1) > abs(s2-1):
            return s2
        else:
            return s1

    class SAVUpdate:
        def __init__(self):
            self.u1 = Function(V)
            self.p1 = Function(Q)
            self.u2 = Function(V)
            self.p2 = Function(Q)
            self.un = Function(V)
            self.pn = Function(Q)
            self.SS = Expression("S", degree=1, S=1.0)
            a1 = inner(u, v)*dx
            self.A1 = assemble(a1)
            self.L1 = inner(self.u1 + self.SS*self.u2, v)*dx
            a2 = inner(p, q)*dx
            self.A2 = assemble(a2)
            self.L2 = inner(self.p1 + self.SS*self.p2, q)*dx

        def solve(self, S):
            self.SS.S = S
            b1 = assemble(self.L1)
            b2 = assemble(self.L2)
            solve(self.A1, self.un.vector(), b1, "bicgstab", "default")
            solve(self.A2, self.pn.vector(), b2, "bicgstab", "default")
            return self.un, self.pn


    sav_solver_1 = SAVSolver1()
    sav_solver_2 = SAVSolver2()
    sav_update = SAVUpdate()

    # Time-stepping
    t = dt
    u0 = Function(V)
    p0 = Function(Q)
    u0.interpolate(u_exact)
    
    while t < T + DOLFIN_EPS:

        # Update boundary condition
        p_exact.t = t
        u_exact.t = t
        f.t = t

        # Solve p0 and u0
        (u1, p1) = sav_solver_1.solve(u0)
        (u2, p2) = sav_solver_2.solve(u0)
        sav_update.p1.assign(p1)
        sav_update.p2.assign(p2)
        sav_update.u1.assign(u1)
        sav_update.u2.assign(u2)
        S = S_solve(u0, u1, u2)
        (un,pn) = sav_update.solve(1)

        # update u0 and t
        u0.assign(un)
        p0.assign(pn)
        t += dt
    

        # Print errors
        print("||u||_2: ", np.sqrt(assemble(inner((u0-u_exact), (u0-u_exact))*dx)))
        print("||p||_2: ", np.sqrt(assemble((p0-p_exact)*(p0-p_exact)*dx)))


for i in range (0, 4):
    for j in range(4):
        SAVPrejectionSolve(pow(3,1+j), solutions[i])