
from fenics import *
from mshr import *
import IB

# 测试一个例子
# 注意points中两个点的顺序，x,y坐标必须从小到大。
points = [Point(0, 0, 0), Point(1, 1, 1)]
seperations = [64, 64, 64]

fluid_mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 1), 64, 64, 64)
solid_mesh = generate_mesh(Sphere(Point(0.5, 0.5, 0.6), 0.2), 20)

Vf = VectorFunctionSpace(fluid_mesh, "P", 2)
Vs = VectorFunctionSpace(solid_mesh, "P", 2)

uf = interpolate(Expression(("x[0]","x[1]","x[2]"),degree=2), Vf)
us = Function(Vs)
IB.interpolate(uf._cpp_object,us._cpp_object, fluid_mesh.hmax(), 1)
File("us.pvd") << us

uf = Function(Vf)
us = interpolate(Expression(("1","2","3"),degree=2), Vs)
IB.interpolate(us._cpp_object,uf._cpp_object, fluid_mesh.hmax(), 4)
File("uf.pvd") << uf

