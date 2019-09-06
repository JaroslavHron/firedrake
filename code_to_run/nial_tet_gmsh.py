#!/usr/bin/env python

import datetime
import time

import sys

from firedrake import *
from firedrake.petsc import PETSc

#np.set_printoptions(precision=2, suppress=False)
info = PETSc.Sys.Print
info_all = PETSc.Sys.syncPrint
distribution_parameters = None


import timeit
my_time_record=0.0

def tic():
   my_time_record=timeit.default_timer()

def toc():
   return(timeit.default_timer()-my_time_record)



#Define domain and mesh
A = 50.0 #velikost krychle
velocity = 10
#NN = 10

#base.init()
#boundary_markers = base.exterior_facets.markers
#print(boundary_markers)

# in firedrake:
# top = 6, bottom = 5, left and right = 1/2, front and back = 3/4
bndry={ "top": 6,
        "bottom": 5,
        "left": 1,
        "right": 2,
        "front": 3,
        "back": 4
        }

# gmsh mesh - 16x16, almost symmetric tests
base = Mesh("mesh_box.msh")

#mh = MeshHierarchy(base, 1)
#base2 = mh[-1]

levels = 2
mh = MeshHierarchy(base, levels, distribution_parameters=distribution_parameters)
mesh = mh[-1]

from A3MconstNewSNES_problem import my_problem

nial_problem = my_problem(A, mesh, bndry, "martensite", extruded=False)


nial_problem.solve(0.0, 6.0/velocity, velocity, True)

chk = DumbCheckpoint("dump_loaded", mode=FILE_CREATE)
f = Function(nial_problem.W, name="all_data")
chk.store(f)
chk.close()
info("Loading saved")

#nial_problem.solve(5.0/velocity, 10.0/velocity, velocity, False)


#nial_problem = my_problem(mesh, "martensite")

#chk = DumbCheckpoint("dump", mode=FILE_READ)
#f = Function(nial_problem.W, name="all_data")
#chk.load(f)

#nial_problem.solve(1.5/velocity, 3.0/velocity, velocity, False)

#time=toc()
info(str(datetime.datetime.now()))
#print("Total time: {}".format(time))
