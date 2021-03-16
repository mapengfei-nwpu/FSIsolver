from fenics import *
from mshr import *
from IBInterpolation import *
from IBMesh import *


# 测试一个例子
# 注意points中两个点的顺序，x,y坐标必须从小到大。
points = [Point(0, 0, 0), Point(1, 1, 0)]
seperations = [64, 64, 0]

regular_mesh = IBMesh(points, seperations)
fluid_mesh = regular_mesh.mesh()
solid_mesh = generate_mesh(Circle(Point(0.6,0.5), 0.2),15)


Vf = VectorFunctionSpace(fluid_mesh, "P", 2)
Vs = VectorFunctionSpace(solid_mesh, "P", 2)

uf = interpolate(Expression(("x[0]","x[1]"),degree=2), Vf)
us = Function(Vs)

# 将坐标移动，然后再进行比较误差
disp_expression = Expression(("x[0]","x[1]"),degree=2)
disp = Function(Vs)
disp.interpolate(disp_expression)

IB = IBInterpolation(regular_mesh, solid_mesh, FunctionSpace(solid_mesh, "P", 2)._cpp_object)

IB.evaluate_current_points(disp._cpp_object)
IB.fluid_to_solid(uf._cpp_object,us._cpp_object)

File("us.pvd") << us

uf = Function(Vf)
us = interpolate(Expression(("1","2"),degree=2), Vs)

IB.solid_to_fluid(uf._cpp_object,us._cpp_object)

File("uf.pvd") << uf
print(assemble(uf[0]*dx))

Vf_t = TensorFunctionSpace(fluid_mesh, "P", 2)
Vs_t = TensorFunctionSpace(solid_mesh, "P", 2)

uf_t = interpolate(Expression((("1","2"),("3","4")),degree=2), Vf_t)
us_t = Function(Vs_t)

IB.fluid_to_solid(uf_t._cpp_object,us_t._cpp_object)
File("us_t.pvd") << us_t

uf_t = Function(Vf_t)
us_t = interpolate(Expression((("sin(100*x[0])","sin(100*x[1])"),("cos(100*x[0])","cos(100*x[1])")),degree=2), Vs_t)

IB.solid_to_fluid(uf_t._cpp_object,us_t._cpp_object)
File("uf_t.pvd") << uf_t
