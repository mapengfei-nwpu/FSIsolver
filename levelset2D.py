from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt  

n = 100
# the width of the margin of the domain.
epsilon = 1/n

class Heaviside(UserExpression):
    def eval(self, values, x):
        level_set = 0.2 - np.sqrt((x[0]-0.6)*(x[0]-0.6) + (x[1]-0.5)*(x[1]-0.5))
        if level_set > epsilon :
            values[0] = 1
        elif level_set < -epsilon:
            values[0] = 0
        else:
            values[0] = 0.5*(1.0 + np.sin(np.pi*level_set/2.0/epsilon))



level_set = Heaviside()
mesh = UnitSquareMesh(100,100)
V = FunctionSpace(mesh, "P", 2)
T = TensorFunctionSpace(mesh, "P", 2)

H = interpolate(level_set, V)
F = project(as_matrix(((h,Constant(0)),(Constant(0),h)),T)


