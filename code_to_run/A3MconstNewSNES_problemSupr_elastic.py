import datetime
import time

import sys
from mpi4py import MPI
import numpy as np

from firedrake import *
from firedrake.petsc import PETSc

np.set_printoptions(precision=2, suppress=False)
distribution_parameters = None

comm = MPI.COMM_WORLD
rank = comm.rank

info = PETSc.Sys.Print

def info_all(s):
   PETSc.Sys.syncPrint("[rank {0}: {1}]".format(rank,s), flush=True, end='\n')

def begin(i,s):
   print("--> [{0}] {1}".format(i,s))

def log(i,s):
   print(" [{0}] {1}".format(i,s))

def end():
   print("--<")

import timeit

my_time_record=0.0

def tic():
   my_time_record=timeit.default_timer()

def toc():
   return(timeit.default_timer()-my_time_record)

from finat import quadrature, point_set

points3 = point_set.PointSet([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
weights3 = [1/24, 1/24, 1/24, 1/24]
rule3 = quadrature.QuadratureRule(points3, weights3)

points2 = point_set.PointSet([[0, 0], [1, 0], [0, 1]])
weights2 = [1/6, 1/6, 1/6]
rule2 = quadrature.QuadratureRule(points2, weights2)


##note that if you have tensor-product cells you would want to provide a tensor product quadrature rule in most cases, use quadrature.TensorProductQuadratureRule([list,of,rules])

#points1 = point_set.PointSet([[0, 0], [1, 0]])
#weights1 = [0.5, 0.5]
#linerule = quadrature.QuadratureRule(points1, weights1)
#quadrule2 = quadrature.TensorProductQuadratureRule([linerule, linerule])
#quadrule3 = quadrature.TensorProductQuadratureRule([linerule, linerule, linerule])

#dx = dx(rule=quadrule3)
dx = dx(rule=rule3)
# dx = dx(degree=2)

ds = ds(rule=rule2)
#ds = ds(degree=2)

#ds_t = ds_t(degree=2)
#ds_b = ds_b(degree=2)
#ds_t = ds_t(rule=quadrule2)
#ds_b = ds_b(rule=quadrule2)

class my_problem(object):
   def __init__(self, A, mesh, bndry, fileName, extruded=False, *args, **kwargs):

      self.mesh = mesh
      self.A=A
      self.bndry=bndry
      
      if not extruded:
         self.ds_t=ds(bndry["top"])
         self.ds_b=ds(bndry["bottom"])
      else:
         self.ds_t=ds_t
         self.ds_b=ds_b
         
         
      #----------DATA---------------
      #hustota energie na rozhrani
      gam = 0.2
      gtw = 0.1
      ell = 0.75
      pi = 3.141592654
      
      phi0 = 0.0
      phi1 = 0.01
      phi2 = 0.01
      phi3 = 0.01
      
      nu0 = 0.01
      nu1 = 0.01
      nu2 = 0.01
      nu3 = 0.01
      
      eps0 = 2.0*ell/pi*(2.0*gam-gtw)
      eps1 = 2.0*ell/pi*gtw
      eps2 = 2.0*ell/pi*gtw
      eps3 = 2.0*ell/pi*gtw
      
      omega0 = 2.0*(2.0*gam-gtw)/(ell*pi)
      omega1 = 2.0*gtw/(ell*pi)
      omega2 = 2.0*gtw/(ell*pi)
      omega3 = 2.0*gtw/(ell*pi)
      
      rho = 500.0
      
      #stiffness tensor NiAl
      c11 = 166.0
      c12 = 136.0
      c44 = 131.0
      
      L1=np.array([[c11,c12,c12,0,0,0],
                   [c12,c11,c12,0,0,0],
                   [c12,c12,c11,0,0,0],
                   [0,0,0,2.0*c44,0,0],
                   [0,0,0,0,2.0*c44,0],
                   [0,0,0,0,0,2.0*c44]])
      
      
      #transformacni Bain tensory NiAl
      alpha = 0.9406
      beta = 1.1302
      
      Logalpha = np.log(0.9406)
      Logbeta = np.log(1.1302)
      
      #--------------DATA -end -------------
      
      #Define function spaces
      self.W = VectorFunctionSpace(self.mesh, "CG", 1, dim=3)
      self.V = VectorFunctionSpace(self.mesh, "CG", 1, dim=3)
      self.Q = FunctionSpace(self.mesh, "CG", 1)
      info("W.dim(): %s" % self.W.dim())
      info_all("num_cells={0}".format(self.mesh.num_cells()))
      
      #Boundary conditions
      bc2 = DirichletBC(self.W.sub(2), Constant(0.0), bndry["bottom"])
      bc3 = DirichletBC(self.W.sub(0), Constant(0.0), [bndry["left"], bndry["right"]])
      bc4 = DirichletBC(self.W.sub(1), Constant(0.0), [bndry["front"], bndry["back"]])

      bcs = [bc2, bc3, bc4]

      #Normal and identity tensor
      n = FacetNormal(self.mesh)
      I = Identity(self.mesh.geometric_dimension())
      
      #Time and timestep
      self.dt = Constant(0.01)
      #t_end = 2.0*8.5/velocity
      #t_end = 10.0/velocity
      #t_end = 24.100
      
      #Def. unknown and test function

      (u_1, u_2, u_3) = split(TestFunction(self.W))
      #(u_1, u_2, u_3, et1_, et2_, et3_) = split(TestFunction(self.W))
      u_ = as_vector([u_1, u_2, u_3])
      
      #Current unknown
      self.w = Function(self.W)
      (u1, u2, u3) = split(self.w)
      #(u1, u2, u3, et1, et2, et3) = split(self.w)
      u = as_vector([u1, u2, u3])
      et1 = 0
      et2 = 0
      et3 = 0
      et0 = 1.0 - et1 - et2 - et3
      
      #previous time step
      self.w0 = Function(self.W)
      (u01, u02, u03) = split(self.w0)
      #(u01, u02, u03, et1p, et2p, et3p) = split(self.w0)
      u0 = as_vector([u01, u02, u03])
      et1p = 0
      et2p = 0
      et3p = 0
      et0p = 1.0 - et1p - et2p - et3p
      
      #Voigtova notace
      def voigt(E):
         e=[E[0,0],E[1,1],E[2,2],sqrt(2)*E[0,1],sqrt(2)*E[0,2],sqrt(2)*E[1,2]]
         f=np.array(e)
         return f
      
      #Formulace ulohy
      #Ft = et0*I + et1*Constant(U1) + et2*Constant(U2) + et3*Constant(U3) #mixing rule
      #Fe = Ff*inv(Ft)
      self.Ff = I + grad(u) #deformation gradient
      self.Ff = variable(self.Ff)
      
      #invFt=as_matrix([[exp(Logalpha*(-et2 - et3) - Logbeta*et1), 0, 0],
                       #[0, exp(Logalpha*(-et1 - et3) - Logbeta*et2), 0],
                       #[0, 0, exp(Logalpha*(-et1 - et2) - Logbeta*et3)]])
      
      Fe = self.Ff
      #Fe = self.Ff*invFt

      CemI = Fe.T*Fe - I
      #He = 0.5*(CemI - 0.5*CemI*CemI + 0.333333333*CemI*CemI*CemI - 0.25*CemI*CemI*CemI*CemI)
      CemIsq = CemI*CemI
      #CemIcub = CemIsq*CemI
      #CemIquad = CemIsq*CemIsq
      #He = 0.5*(CemI - 0.5*CemIsq + CemIcub/3.0 - 0.25*CemIquad + 0.2*CemIquad*CemI - (CemIcub*CemIcub)/6.0 + (CemIquad*CemIcub)/7.0 - 0.125*CemIquad*CemIquad)
      #He = 0.5*(CemI - 0.5*CemIsq + CemIcub/3.0 - 0.25*CemIsq*CemIsq + 0.2*CemIcub*CemIsq - (CemIcub*CemIcub)/6.0)
      He = 0.5*((CemI + 0.5*CemIsq)*inv(I + CemI + CemIsq/6.0))
      #He = 0.5*(CemI - 0.5*CemI*CemI + (CemI*CemI*CemI)/3.0)
      e = as_vector(voigt(He))

      #Ee = 0.5*(Fe.T*Fe - I)
      #e = as_vector(voigt(Ee))

      L = Constant(L1)
      self.FB = 0.5*inner(e,L*e) #+ et0*phi0 + et1*phi1 + et2*phi2 + et3*phi3 # det(Ft) missing
      #FI = eps0*inner(grad(et0),grad(et0)) + omega0*et0*(1.0-et0) +eps1*inner(grad(et1),grad(et1)) + omega1*et1*(1.0-et1) + eps2*inner(grad(et2),grad(et2)) + omega2*et2*(1.0-et2) + eps3*inner(grad(et3),grad(et3)) + omega3*et3*(1.0-et3)
      
      F = self.FB #+ FI
#      Diss = 0.5*(nu0*(et0-et0p)*(et0-et0p) + nu1*(et1-et1p)*(et1-et1p) + nu2*(et2-et2p)*(et2-et2p) + nu3*(et3-et3p)*(et3-et3p)) / self.dt
      Pi = F*dx
      #Pi = (F + Diss)*dx

      #pen0 = conditional(lt(et0, 0.0), rho, 0.0)*et0*et0
      #pen1 = conditional(lt(et1, 0.0), rho, 0.0)*et1*et1
      #pen2 = conditional(lt(et2, 0.0), rho, 0.0)*et2*et2
      #pen3 = conditional(lt(et3, 0.0), rho, 0.0)*et3*et3
      
      #pen = (pen0 + pen1 + pen2 + pen3)*dx
      
      self.Rc = 50.0
      rhotwo = 500.0
      x = SpatialCoordinate(self.mesh)
      self.posz = Constant(150.0)
      ###gap = Expression("sqrt((posx-x[0]-u[0])*(posx-x[0]-u[0])+(posy-x[1]-u[1])*(posy-x[1]-u[1])+(posz-x[2]-u[2])*(posz-x[2]-u[2])) - Rc", posx = 0.0, posy = 0.0, posz = A + Rc, degree = 2)
      ###gap = Expression("sqrt((-x[0]-u[0])*(-x[0]-u[0])+(-x[1]-u[1])*(-x[1]-u[1])+(posz-x[2]-u[2])*(posz-x[2]-u[2])) - Rc", posz = 35.0, Rc = Rc, degree = 2)
      gap = sqrt((-x[0]-u[0])*(-x[0]-u[0])+(-x[1]-u[1])*(-x[1]-u[1])+(self.posz-x[2]-u[2])*(self.posz-x[2]-u[2])) - self.Rc
      #gap = (posz-x[2]-u[2]) # plane gap
      
      penpot = conditional(lt(gap, 0.0), rhotwo, 0.0)*gap*gap
      contact = penpot*self.ds_t
      
      #Derivatives
      self.R = derivative(Pi + contact, self.w)
      J = derivative(self.R, self.w)
      
      self.pvd = File("results_%s/state.pvd" % fileName)
      
      self.problem = NonlinearVariationalProblem(self.R, self.w, bcs=bcs, J=J)
      self.bcs = bcs
      lu = {"mat_type": "baij",
            "snes_type": "newtonls",
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_max_it": 12,
            "snes_rtol": 1e-11,
            "snes_atol": 5e-10,
            "snes_linesearch_type": "basic",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"}
      
      gamg = {"mat_type": "aij",
              "snes_type": "newtonls",
              "snes_monitor": None,
              "snes_max_it": 13,
              "snes_rtol": 1e-11,
              "snes_atol": 5e-10,
              "snes_linesearch_type": "basic",
              "ksp_type": "fgmres",
              "ksp_max_it": 500,
              "ksp_converged_reason": None,
              "pc_type": "gamg"}

      cgjacobi = {"mat_type": "aij",
                  "snes_type": "newtonls",
                  "snes_monitor": None,
                  "snes_max_it": 13,
                  "snes_rtol": 1e-11,
                  "snes_atol": 1e-8,
                  "snes_linesearch_type": "basic",
                  "ksp_type": "cg",
                  "ksp_rtol": 1e-6,
                  "ksp_atol": 5e-12,
                  "ksp_max_it": 1000,
                  "ksp_converged_reason": None,
                  "pc_type": "jacobi"}
      
      mg = {"mat_type": "baij",
	    #"snes_type": "vinewtonssls",
            "snes_type": "newtonls",
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_max_it": 13,
            "snes_rtol": 1e-11,
            "snes_atol": 1e-8,
            "snes_linesearch_type": "basic",
            #"snes_linesearch_type": "l2",
            "snes_linesearch_maxstep": 1.0,
            "snes_linesearch_monitor": None,
            "ksp_type": "fgmres",
            #"ksp_monitor": None,
            #"mg_coarse_ksp_monitor": None,
            #"mg_levels_ksp_monitor": None,
            "ksp_max_it": 150,
            #"ksp_type": "cg",
            #"ksp_max_it": 600,
            "ksp_rtol": 1e-6,
            "ksp_atol": 5e-12,
            "ksp_gmres_restart": 150,
            "ksp_converged_reason": None,
            #"ksp_monitor_true_residual": None,
            "pc_type": "mg",
            "mg_coarse_ksp_type": "preonly",
            "mg_coarse_pc_type": "lu",
            "mg_coarse_pc_factor_mat_solver_type": "mumps",
            "mg_coarse_mat_mumps_icntl_14": 1000,
            #"mg_coarse_mat_mumps_icntl_14": 500,
            #"mg_levels_ksp_type": "chebyshev",
            "mg_levels_ksp_type": "gmres",
            "mg_levels_ksp_max_it": 4,
            "mg_levels_ksp_convergence_test": "skip",
            #"mg_levels_ksp_monitor_true_residual": None,
            "mg_levels_ksp_norm_type": "unpreconditioned",
            "mg_levels_pc_type": "pbjacobi"
      }
      
      self.sp = lu
      self.solver  = NonlinearVariationalSolver(self.problem, solver_parameters=self.sp)
      
   def solve(self, t_init, t_end, velocity, loading):
      #self.its, self.ok, _ = self.solver.solve()
      #return(self.its,self.ok)

      A=self.A
      optimal_it = 9.0
      dt_max = 0.25 / velocity
      self.dt.assign(0.01)
      info("dt = {}".format(float(self.dt)))
      
      uviz = Function(self.V, name="displacement")
      #etw1 = Function(self.Q, name="eta1")
      #etw2 = Function(self.Q, name="eta2")
      #etw3 = Function(self.Q, name="eta3")
      
      (uw1, uw2, uw3) = split(self.w)
      #(uw1, uw2, uw3, etw1_, etw2_, etw3_) = split(self.w)
      uviz.interpolate(as_vector([uw1, uw2, uw3]))
      #etw1.interpolate(etw1_)
      #etw2.interpolate(etw2_)
      #etw3.interpolate(etw3_)
      self.pvd.write(uviz, time=t_init)
      #self.pvd.write(uviz, etw1, etw2, etw3, time=t_init)
      
      #tic()
      # Time-stepping
      self.t = float(t_init + self.dt)
      ii = 0
     
      failed = False
     
      while self.t < t_end + dt_max - 0.0001:
         if self.t < 2.0:
            self.posz.assign(float(self.Rc + A - velocity*self.t))
         elif self.t < 2.2:
            self.posz.assign(float(self.Rc + A - 2.0*velocity))
         else:
            self.posz.assign(float(self.Rc + A - 2.0*velocity + velocity*(self.t - 2.2)))

         #if(loading):
            #self.posz.assign(float(self.Rc + A - velocity*self.t))
         #else:
            #self.posz.assign(float(self.Rc + A + velocity*(self.t - 2.0*t_init)))

         info("t = {}".format(self.t))

         # Compute
         try:
            solvestart = time.time()
            self.solver.solve()
            solveend = time.time()
            its = self.solver.snes.getIterationNumber()
            reason = self.solver.snes.getConvergedReason()
            ok = (reason > 0)
            failed = False
         except ConvergenceError:
            info("FAILED")
            its = self.solver.snes.getIterationNumber()
            reason = self.solver.snes.getConvergedReason()
            ok = (reason > 0)
            failed = True
            self.solver  = NonlinearVariationalSolver(self.problem, solver_parameters=self.sp)
            
         #if (ok == 0 or reason == 7):
         if (ok == 0):
            info("Step back!")

            #r = Function(self.W)
            #assemble(self.R, tensor=r)
            #for bc in self.bcs : bc.apply(r, u=self.w)
            #(ru1, ru2, ru3, ret1, ret2, ret3) = split(r)
            #info("residua |ru1|, |ru2|, |ru3|, |ret1|, |ret2|, |ret3| = {{{0:f}, {1:f}, {2:f}, {3:f}, {4:f}, {5:f}}},".format(norm(ru1), norm(ru2), norm(ru3), norm(ret1), norm(ret2), norm(ret3)))

            self.t -= float(self.dt)/2.0
            self.dt.assign(float(self.dt)/2.0)
            info("dt = {}".format(float(self.dt)))
            self.w.assign(self.w0)
            failed = True
            continue

         info("Solver time taken: %s seconds" % (solveend - solvestart))

         # Alternative force calculation
         w_=Function(self.W)
         DirichletBC(self.W.sub(2), 1.0, self.bndry["bottom"]).apply(w_)
         force = assemble(action(self.R, w_))
         P = diff(self.FB, self.Ff)
         force2 = assemble(-P[2,2]*self.ds_b)
         
         info("indentation, force, force2 = {{{0:f}, {1:f}, {2:f}}},".format(float(A + self.Rc - self.posz), force, force2))

         if not failed:
            if its <= optimal_it:
               self.dt.assign(min(dt_max, float(self.dt)*min(1.6,optimal_it/(its+0.001))))
            else:
               self.dt.assign(float(self.dt)*optimal_it/its)

         ii = ii + 1
         
         # Extract solutions:
         if (ii % 4 == 0):
            (uw1, uw2, uw3) = split(self.w)
            #(uw1, uw2, uw3, etw1_, etw2_, etw3_) = split(self.w)
            uviz.interpolate(as_vector([uw1, uw2, uw3]))
            #etw1.interpolate(etw1_)
            #etw2.interpolate(etw2_)
            #etw3.interpolate(etw3_)
            self.pvd.write(uviz, time=self.t)
            #self.pvd.write(uviz, etw1, etw2, etw3, time=self.t)

         self.w0.assign(self.w)

         if self.t == t_end:
            break

         # Move to next time step
         #self.w0.assign(self.w)
         self.t += float(self.dt)
         if self.t > t_end:
            self.dt.assign(t_end - self.t +float(self.dt))
            self.t = t_end

         info("dt = {}".format(float(self.dt)))
         info(str(datetime.datetime.now()))



