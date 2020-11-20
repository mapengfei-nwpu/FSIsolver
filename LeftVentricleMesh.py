# triangulation
from fenics import *
from mshr import *

mesh_size = 50
big = Ellipsoid(Point(0,0,0),1.0,1.0,2.0)
small = Ellipsoid(Point(0,0,0),0.7,0.7,1.7)
gaizi = Box(Point(-1.1,-1.1,0.5),Point(1.1,1.1,2.1))
domain = big-small-gaizi
mesh = generate_mesh(domain, mesh_size)

# mark the boundary
mesh_function = MeshFunction("size_t", mesh, 2, value = 0)

class VentricleBase(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2],0.5)


class VentricleWall(SubDomain):
    def inside(self, x, on_boundary):
        temp = (x[0]*x[0] + x[1]*x[1])/0.7/0.7 + x[2]*x[2]/1.7/1.7
        return on_boundary and temp < 1.001


venbase = VentricleBase()
venbase.mark(mesh_function, 1)
venwall = VentricleWall()
venwall.mark(mesh_function, 2)

if __name__ == '__main__' :
    # visulize mesh_function
    V = FunctionSpace(mesh, "P", 1)
    U = VectorFunctionSpace(mesh, "P", 1)

    bcs = [DirichletBC(V, Constant(100), mesh_function, 1),
        DirichletBC(V, Constant(10000), mesh_function, 2)]

    bcs_component = [DirichletBC(U.sub(0), Constant(10000), mesh_function, 2)]

    v = Function(V)
    u = Function(U)

    for bc in bcs:
        bc.apply(v.vector())

    for bc in bcs_component:
        bc.apply(u.vector())

    File("u.pvd") << u
    File("v.pvd") << v