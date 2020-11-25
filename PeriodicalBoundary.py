
from fenics import *


# min = [0.0, 0.0, 0.0]
# max = [1.0, 1.0, 1.0]

min = [-20.0, -20.0, -30.0]
max = [ 20.0,  20.0,  10.0]

class PeriodicalBoundary(SubDomain):

    # The boundary is left, down, back
    def inside(self, x, on_boundary):
        if on_boundary:
            if near(x[0], min[0], eps = 1E-5) :
                return True
            if near(x[1], min[1], eps = 1E-5) :
                return True
            if near(x[2], min[2], eps = 1E-5) :
                return True
        else :
            return False

    # Map the right, up, front to the left, down, back respectively
    def map(self, x, y):
        if near(x[0], max[0], eps = 1E-5):
            y[0] = x[0] - max[0]
        else :
            y[0] = x[0]
        
        if near(x[1], max[1], eps = 1E-5):
            y[1] = x[1] - max[1]
        else :
            y[1] = x[1]        
        
        if near(x[2], max[2], eps = 1E-5):
            y[2] = x[2] - max[2]
        else :
            y[2] = x[2]




periodic_boundary = PeriodicalBoundary(1E-5)

if __name__ == "__main__":
    periodic_boundary = PeriodicalBoundary(1E-5)
    mesh = UnitCubeMesh(10,10,10)
    V = FunctionSpace(mesh, "P", 1, constrained_domain=periodic_boundary)
    u = Function(V)
    
    print("size of u.vector() : ", u.vector().size())
    print("number of coordinates : ", mesh.coordinates().size)

    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], min[0], eps = 1E-5)
    
    class DownBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], min[1], eps = 1E-5)
    
    class BackBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[2], min[2], eps = 1E-5)

    left_boundary = LeftBoundary()
    down_boundary = DownBoundary()
    back_boundary = BackBoundary()

    bc1 = DirichletBC(V, 1, left_boundary)
    bc2 = DirichletBC(V, 2, down_boundary)
    bc3 = DirichletBC(V, 3, back_boundary)

    bc1.apply(u.vector())
    bc2.apply(u.vector())
    bc3.apply(u.vector())

    File("u.pvd") << u

