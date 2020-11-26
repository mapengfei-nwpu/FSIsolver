from .ParticleInterpolationPybind import interpolate as PI_interpolate
from fenics import Function

class ParticleInterpolation:
    def __init__(self, fluid_mesh, solid_mesh, disp):
        self.disp = Function(disp.function_space())
        self.disp.assign(disp)
        self.fluid_mesh = fluid_mesh
        self.solid_mesh = solid_mesh
    def fluid_to_solid(self, fluid, solid):
        PI_interpolate(fluid._cpp_object, self.disp._cpp_object, solid._cpp_object, self.fluid_mesh.hmax(), 4, False)
    def solid_to_fluid(self, fluid, solid):
        PI_interpolate(solid._cpp_object, self.disp._cpp_object, fluid._cpp_object, self.fluid_mesh.hmax(), 6, True)
    def evaluate_current_points(self, disp):
        self.disp.vector()[:] = disp.vector()[:]

if __name__ == '__main__' :
    from fenics import *
    from mshr import *
    # 测试一个例子
    # 注意points中两个点的顺序，x,y坐标必须从小到大。

    fluid_mesh = BoxMesh(Point(0, 0, 0), Point(1, 1, 1), 32, 32, 32)
    solid_mesh = generate_mesh(Sphere(Point(0.5, 0.5, 0.6), 0.2), 20)


    Vf = VectorFunctionSpace(fluid_mesh, "P", 2)
    Vs = VectorFunctionSpace(solid_mesh, "P", 2)

    # 将坐标移动，然后再进行比较误差
    disp_expression = Expression(("x[0]","x[1]","x[2]+0.1"),degree=2)
    disp = Function(Vs)
    disp.interpolate(disp_expression)

    IB = ParticleInterpolation(fluid_mesh, solid_mesh, disp)

    uf = interpolate(Expression(("x[0]","x[1]","x[2]"),degree=2), Vf)
    us = Function(Vs)
    IB.evaluate_current_points(disp)
    IB.fluid_to_solid(uf,us)
    File("us.pvd") << us

    uf = Function(Vf)
    us = interpolate(Expression(("1","2","3"),degree=2), Vs)
    IB.evaluate_current_points(disp)

    import time
    time_start=time.time()
    IB.solid_to_fluid(uf,us)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')

    File("uf.pvd") << uf
