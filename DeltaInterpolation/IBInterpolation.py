from fenics import *
from IBInterpolation import *
from IBMesh import *
import numpy as np
# 测试一个例子
# 注意points中两个点的顺序，现在x,y坐标必须从小到大。
points = [Point(-2, -2, 0), Point(2, 2, 0)]
seperations = [80, 80, 0]

regular_mesh = IBMesh(points, seperations)
fluid_mesh = regular_mesh.mesh()
solid_mesh = UnitSquareMesh(10,10)

Vf = VectorFunctionSpace(fluid_mesh, "P", 2)
Vs = VectorFunctionSpace(solid_mesh, "P", 2)

uf = project(Expression(("x[0]*x[0]","x[1]*x[1]"),degree=2), Vf)
us = Function(Vs)

# 将坐标移动，然后再进行比较误差
disp_expression = Expression(("x[0]+0.1","x[1]+0.1"),degree=2)
disp = project(disp_expression,Vs)

IB = IBInterpolation(regular_mesh, solid_mesh, us._cpp_object)
IB.evaluate_current_points(disp._cpp_object)
IB.fluid_to_solid(uf._cpp_object, us._cpp_object)

n = 100
rd = np.random.rand(2*n)*0.8+0.1
for i in range(n):
    result = us(rd[i*2],rd[i*2+1])
    print(result[0] - (rd[i*2]+0.1)*(rd[i*2]+0.1))
    print(result[1] - (rd[i*2+1]+0.1)*(rd[i*2+1]+0.1))


us = project(Expression(("1","1"),degree=2), Vs)
uf = Function(Vf)

IB.solid_to_fluid(uf._cpp_object, us._cpp_object)

rd = np.random.rand(2*n)*0.8+0.2
# 测试。需要把点移回到固体点再计算。
for i in range(n):
    result = uf(rd[i*2],rd[i*2+1])
    print(result[0] - (rd[i*2]-0.1)*(rd[i*2]-0.1))
    print(result[1] - (rd[i*2+1]-0.1)*(rd[i*2+1]-0.1))


File("uf.pvd") << uf
