from fenics import *
# this function is used to calculate the out normal of a
# given mesh. This code comes from the discussion on
# https://fenicsproject.discourse.group/t/how-to-plot-no
# rmal-unit-vector-of-faces-in-a-2d-mesh/3912
def out_normal(mesh):
    n = FacetNormal(mesh)
    V = VectorFunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u,v)*ds
    l = inner(n,v)*ds
    A = assemble(a, keep_diagonal=True)
    L = assemble(l)
    A.ident_zeros()
    nh = Function(V)
    solve(A, nh.vector(), L)
    return nh


# 3. Pressure on the inner wall.
def wall_pressure(displace, force, boundary, marker):
    
    V = force.function_space()
    C = 10.0
    mesh = V.mesh()

    # moveback = original position - current position 
    # moveto   = -moveback
    moveback = interpolate(Expression(("x[0]", "x[1]", "x[2]"), degree=1), V)
    moveto = Function(V)
    moveback_vector = moveback.vector()
    moveback_vector -= displace.vector()
    moveto_vector = moveto.vector()
    moveto_vector -= moveback.vector()

    ALE.move(mesh, moveto)
    File("to_mesh.pvd") << mesh
    n = out_normal(mesh)
    ALE.move(mesh, moveback)
    File("back_mesh.pvd") << mesh

    # get a unit function.
    # unit = Function(V)
    # unit.vector().vec().shift(1)

    # get a zero function.
    # zero = Function(V)

    # get the pressure on normal direction.
    n_vector = n.vector()
    n_vector *= -C

    u = Function(V)
    bc = DirichletBC(V, n, boundary, marker)
    bc.apply(u.vector())
    
    force.vector().axpy(1.0,u.vector())


# 2. Lagrangian multiplier on base
def base_constraint(displace, force, boundary, marker):

    V = force.function_space()
    # Zhang Ruihang told me, in the simulation of heart valve,
    # the penalty is 50kPa when the pressure of blood is about 10kPa.
    # which is 5*10^6 dyn/cm^2
    penalty = 500                    
    move = interpolate(Expression(("x[0]", "x[1]", "x[2]"), degree=2), V)
    move_vector = move.vector()
    move_vector -= displace.vector()
    # f = 1e4*(x0-xt)
    
    z_direction = False # constraint can be in z direction or in x,y,z directions

    if z_direction :
        # extract the z component of displacement
        z = move.sub(2, deepcopy = True)
        # apply the boundary condition
        z_vector = z.vector()
        z_vector *= penalty
        bc = DirichletBC(V.sub(2), z, boundary, marker)
    else :
        p = Function(V)
        p.assign(move)
        p_vector = p.vector()  # p_vector is a reference to the vector
        p_vector *= penalty
        bc = DirichletBC(V, p, boundary, marker)

    u = Function(V)
    bc.apply(u.vector())    
    force.vector().axpy(1.0,u.vector())


def apply_boundary_conditions(disp, force, solid_boundary, marker_base, marker_wall):
    wall_pressure(disp, force, solid_boundary, marker_wall)
    base_constraint(disp, force, solid_boundary, marker_base)


if __name__ == "__main__":
    from LeftVentricleMesh     import mesh          as solid_mesh
    from LeftVentricleMesh     import mesh_function as solid_boundary
    V = VectorFunctionSpace(solid_mesh, "P", 2)
    disp = interpolate(Expression(("x[0]+1", "x[1]+1", "x[2]+1"), degree=2), V)
    force = interpolate(Expression(("10000", "10000", "10000"), degree=2), V)

    marker_base = 1
    marker_wall = 2

    # wall_pressure(disp, force, solid_boundary, marker_wall)
    File("wall_pressure.pvd") << force

    # base_constraint(disp, force, solid_boundary, marker_base)
    File("base_constraint.pvd") << force

    apply_boundary_conditions(disp, force, solid_boundary, marker_base, marker_wall)
    File("disp.pvd") << disp