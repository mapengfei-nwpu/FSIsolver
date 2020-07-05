from fenics import *
import IBMesh

# 测试一个例子
# 注意points中两个点的顺序，x,y坐标必须从小到大。
points = [Point(0, 0, 0), Point(1, 1, 0)]
seperations = [100, 100, 0]
regular_mesh = IBMesh.IBMesh(points, seperations)
mesh = regular_mesh.mesh()
adjacents = regular_mesh.hash(Point(0.1,0.1))

# 测试一组随机点
import numpy as np
from mpi4py import MPI
n = 10
total = 0
if MPI.COMM_WORLD.Get_rank() == 0:
    r = np.random.rand(2*n)
else:
    r = None

r = MPI.COMM_WORLD.bcast(r, root=0)
f = open(str(MPI.COMM_WORLD.Get_rank())+"out.txt", "w")
for i in range(n):
    point = Point(r[2*i],r[2*i+1])
    if regular_mesh.find_cell(point, cell) :
        total += 1
        global_index = cell.global_index()
        map_to_local = regular_mesh.map(global_index)
        print("checking... ",file=f)
        print("contains: ", cell.contains(point),file=f)
        print("hash : ", int(global_index/2), "and", regular_mesh.hash(point), file=f)
        print("local_index : ", cell.index(), "and", map_to_local[1], file = f)
        print("mpi_rank : ", MPI.COMM_WORLD.Get_rank(), "and", map_to_local[0], file=f)


print(total,file=f)
f.close()

# mkdir build && cd build
# cp ../IBMesh.py .
# cmake ..
# make
# mpirun -np 8 python3 IBMesh.py