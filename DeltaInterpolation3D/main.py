
from fenics import *
from mshr import *
from IB import *


# 测试一个例子
# 注意points中两个点的顺序，x,y坐标必须从小到大。
points = [Point(0, 0, 0), Point(1, 1, 1)]
seperations = [64, 64, 64]

regular_mesh = IBMesh(points, seperations)
fluid_mesh = regular_mesh.mesh()
solid_mesh = generate_mesh(Sphere(Point(0.5,0.5,0.6), 0.2),20)


Vf = VectorFunctionSpace(fluid_mesh, "P", 2)
Vs = VectorFunctionSpace(solid_mesh, "P", 2)

uf = interpolate(Expression(("x[0]","x[1]","x[2]"),degree=2), Vf)
us = Function(Vs)

# 将坐标移动，然后再进行比较误差
disp_expression = Expression(("x[0]","x[1]","x[2]+0.1"),degree=2)
disp = Function(Vs)
disp.interpolate(disp_expression)

IB = DeltaInterpolation(regular_mesh, solid_mesh, us._cpp_object)

IB.evaluate_current_points(disp._cpp_object)
IB.fluid_to_solid(uf._cpp_object,us._cpp_object)

File("us.pvd") << us

uf = Function(Vf)
us = interpolate(Expression(("1","2","3"),degree=2), Vs)

IB.solid_to_fluid(uf._cpp_object,us._cpp_object)
IB.solid_to_fluid(uf._cpp_object,us._cpp_object)

File("uf.pvd") << uf
print(assemble(uf[0]*dx))
