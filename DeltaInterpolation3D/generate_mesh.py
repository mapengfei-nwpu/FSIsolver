from fenics import *
from mshr import *

mesh = generate_mesh(Sphere(Point(0.5,0.5,0.5), 0.1),10)
print(mesh.num_entities(3))
File("sphere.xml.gz") << mesh
