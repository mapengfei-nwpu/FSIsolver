
from fenics import *
from mshr import *
import IB

# 测试一个例子
# 注意points中两个点的顺序，x,y坐标必须从小到大。

fluid_mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 1), 32, 32, 32)
solid_mesh = generate_mesh(Sphere(Point(0.5, 0.5, 0.5), 0.2), 20)

Vf = VectorFunctionSpace(fluid_mesh, "P", 2)
Vs = VectorFunctionSpace(solid_mesh, "P", 2)

# 将坐标移动，然后再进行比较误差
disp_expression = Expression(("x[0]+0.1","x[1]+0.1","x[2]+0.1"),degree=2)
disp = Function(Vs)
disp.interpolate(disp_expression)

uf = interpolate(Expression(("x[0]","x[1]","x[2]"),degree=2), Vf)
us = Function(Vs)
IB.interpolate(uf._cpp_object, disp._cpp_object, us._cpp_object, fluid_mesh.hmax(), 1, False)
File("us.pvd") << us

uf = Function(Vf)
us = interpolate(Expression(("1","2","3"),degree=2), Vs)
IB.interpolate(us._cpp_object, disp._cpp_object, uf._cpp_object, fluid_mesh.hmax(), 4, True)
File("uf.pvd") << uf

