#!/usr/bin/env python

import datetime
import time

import sys
import os

#import petsc4py
#petsc4py.init(sys.argv)

from firedrake import *
from firedrake.petsc import PETSc

#np.set_printoptions(precision=2, suppress=False)
info = PETSc.Sys.Print
info_all = PETSc.Sys.syncPrint
distribution_parameters = None

opts = PETSc.Options()
opts['options_left'] = True
opts['options_monitor'] = True
opts['ksp_monitor'] = True
opts['snes_monitor'] = True
opts['snes_view'] = True
opts['snes_linesearch_monitor'] = True
#opts.setFromOptions()

import timeit
my_time_record=0.0

def tic():
   my_time_record=timeit.default_timer()

def toc():
   return(timeit.default_timer()-my_time_record)



#Define domain and mesh
A = 100.0 #velikost krychle
velocity = 10.0
NN = 10

# gmsh mesh - 8x8, almost symmetric tests
base = Mesh("mesh_box_100.msh")

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

levels = 2

mh = MeshHierarchy(base, levels)
mesh = mh[-1]

from finat import quadrature, point_set
points3 = point_set.PointSet([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
weights3 = [1/24, 1/24, 1/24, 1/24]
rule3 = quadrature.QuadratureRule(points3, weights3)

points2 = point_set.PointSet([[0, 0], [1, 0], [0, 1]])
weights2 = [1/6, 1/6, 1/6]
rule2 = quadrature.QuadratureRule(points2, weights2)


from A3Elast import my_problem

fileName="results_{0}".format(os.path.splitext(__file__)[0])

#nial_problem = my_problem(A, mesh, bndry, fileName, qr3d=rule3, qr2d=rule2, extruded=False)
nial_problem = my_problem(A, mesh, bndry, fileName, extruded=False)

nial_problem.solver.snes.view()
nial_problem.solve(0.0, 2.0, velocity)
info("=====================================================================+++")
nial_problem.solve(2.0, 4.0, -velocity)

info(str(datetime.datetime.now()))
#print("Total time: {}".format(time))
