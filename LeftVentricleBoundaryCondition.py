from fenics import *
from LeftVentricleMesh     import mesh          as solid_mesh
from LeftVentricleMesh     import mesh_function as solid_boundary
from PassiveLeftVentricle  import first_PK_stress


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

#TODO: 1.Active stress tensor.
#TODO: 2.Multiplier on base at the z direction.
#TODO: 3.Pressure on the inner wall.
#TODO: 4.Unit conversion: mesh size, viscous coefficient, pressure.


if __name__ == "__main__":
    n = out_normal(solid_mesh)                                              # 3.calculation of the out normal direction 
    File("nh_1.pvd") << n                                                   
    Q = FunctionSpace(solid_mesh, "P", 2)
    V = VectorFunctionSpace(solid_mesh, "P", 2)
    bc = DirichletBC(V, n, solid_boundary, 2)                               # 3.set it as the boundary condition
    u = Function(V)
    bc.apply(u.vector())                                                    # 3.apply the pressure on the wall on the out normal direction
    File("u.pvd") << u
    z = n.sub(0,deepcopy=True)                                              # 2.Extract the third component of displacement as a lagrangian multiplier. 
    bc_component = DirichletBC(V.sub(2), n[0], solid_boundary, 2)           # 2.multiplier should be implied on the third component of interactive force.