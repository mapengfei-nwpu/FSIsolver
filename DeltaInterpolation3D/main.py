
from fenics import *
from mshr import *
from IB import *


# 测试一个例子
# 注意points中两个点的顺序，x,y坐标必须从小到大。
points = [Point(0, 0, 0), Point(1, 1, 1)]
seperations = [100, 100, 100]

regular_mesh = IBMesh(points, seperations)
fluid_mesh = regular_mesh.mesh()
solid_mesh = generate_mesh(Sphere(Point(0.5,0.5,0.5), 0.3),20)


Vf = VectorFunctionSpace(fluid_mesh, "P", 2)
Vs = VectorFunctionSpace(solid_mesh, "P", 2)

uf = interpolate(Expression(("x[0]","x[1]","x[2]"),degree=2), Vf)
us = Function(Vs)

# 将坐标移动，然后再进行比较误差
disp_expression = Expression(("x[0]+0.1","x[1]+0.1","x[2]+0.1"),degree=2)
disp = Function(Vs)
disp.interpolate(disp_expression)

IB = DeltaInterpolation(regular_mesh, solid_mesh, us._cpp_object)

IB.evaluate_current_points(disp._cpp_object)
IB.fluid_to_solid(uf._cpp_object,us._cpp_object)

File("us.pvd") << us

uf = Function(Vf)
us = interpolate(Expression(("x[0]*x[0]","x[1]*x[1]","x[2]*x[2]"),degree=2), Vs)

IB.solid_to_fluid(uf._cpp_object,us._cpp_object)

File("uf.pvd") << uf

import numpy as np
test_data = np.random.rand(300)
for i in range(300):
    if (test_data[3*i]-0.5)*(test_data[3*i]-0.5)+(test_data[3*i+1]-0.5)*(test_data[3*i+1]-0.5)+(test_data[3*i+2]-0.5)*(test_data[3*i+2]-0.5)<0.09:
        a = uf(test_data[3*i], test_data[3*i+1], test_data[3*i+2])
        print(a)
        print(test_data[3*i], test_data[3*i+1], test_data[3*i+2])

