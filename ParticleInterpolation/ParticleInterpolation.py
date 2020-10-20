from ParticleInterpolationPybind import interpolate as PI_interpolate
from fenics import Function

class ParticleInterpolation:
    def __init__(self, fluid_mesh, solid_mesh, disp):
        self.disp = Function(disp.function_space())
        self.disp.assign(disp)
        self.fluid_mesh = fluid_mesh
        self.solid_mesh = solid_mesh
    def fluid_to_solid(self, fluid, solid):
        PI_interpolate(fluid._cpp_object, self.disp._cpp_object, solid._cpp_object, self.fluid_mesh.hmax(), 1, False)
    def solid_to_fluid(self, fluid, solid):
        PI_interpolate(solid._cpp_object, self.disp._cpp_object, fluid._cpp_object, self.fluid_mesh.hmax(), 4, True)
    def evaluate_current_points(self, disp):
        self.disp.vector()[:] = disp.vector()[:]
