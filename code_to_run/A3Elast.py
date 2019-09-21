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

class my_problem(object):
   def __init__(self, A, mesh, bndry, fileName, qr3d=None, qr2d=None, extruded=False, *args, **kwargs):

      self.mesh = mesh
      self.A=A
      self.bndry=bndry

      global dx,ds,ds_t,ds_b
      
      deg=2
      if qr3d is None:
         dx = dx(degree=deg)  
      else:
         dx = dx(rule=qr3d)
      
      if qr2d is None:
         ds = ds(degree=deg)  
         ds_t = ds_t(degree=deg)
         ds_b = ds_b(degree=deg)         
      else:
         ds = ds(rule=qr2d)
         ds_t = ds_t(rule=qr2d)
         ds_b = ds_b(rule=qr2d)         
      
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
      SCG2=FiniteElement("CG", self.mesh.ufl_cell(), degree=2)  #, variant='spectral')
      SCG1=FiniteElement("CG", self.mesh.ufl_cell(), degree=1)  #, variant='spectral')
      #self.W = MixedFunctionSpace([SCG1, SCG1, SCG1, SCG1, SCG1, SCG1])
      self.W = VectorFunctionSpace(self.mesh, SCG1)
      W=self.W
      self.V = VectorFunctionSpace(self.mesh, "CG", 1)
      self.Q = FunctionSpace(self.mesh, "CG", 1)
      info("W.dim(): %s" % self.W.dim())
      info_all("num_cells={0}".format(self.mesh.num_cells()))
      
      #Boundary conditions
      bc2 = DirichletBC(self.W.sub(2), Constant(0.0), bndry["bottom"])
      bc2a = DirichletBC(self.W.sub(0), Constant(0.0), bndry["bottom"])
      bc2b = DirichletBC(self.W.sub(1), Constant(0.0), bndry["bottom"])
      bc3 = DirichletBC(self.W.sub(0), Constant(0.0), [bndry["left"], bndry["right"]])
      bc4 = DirichletBC(self.W.sub(1), Constant(0.0), [bndry["front"], bndry["back"]])
      
      self.bcs = [bc2, bc3, bc4]
      
      #Normal and identity tensor
      n = FacetNormal(self.mesh)
      I = Identity(self.mesh.geometric_dimension())
      
      #Time and timestep
      self.dt = Constant(0.01)
      #t_end = 2.0*8.5/velocity
      #t_end = 10.0/velocity
      #t_end = 24.100
      
      #Def. unknown and test function
      u_ = TestFunction(self.W)
      
      #Current unknown
      self.w = Function(self.W)
      u = self.w

      et1=Constant(0.0)
      et2=Constant(0.0)
      et3=Constant(0.0)
      
      et0 = 1.0 - et1 - et2 - et3
      
      #previous time step
      self.w0 = Function(self.W)
      u0 = self.w0

      et1p=Constant(0.0)
      et2p=Constant(0.0)
      et3p=Constant(0.0)
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
      
      invFt=as_matrix([[exp(Logalpha*(-et2 - et3) - Logbeta*et1), 0, 0],
                       [0, exp(Logalpha*(-et1 - et3) - Logbeta*et2), 0],
                       [0, 0, exp(Logalpha*(-et1 - et2) - Logbeta*et3)]])
      
      Fe = self.Ff*invFt

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
      self.FB = 0.5*inner(e,L*e) + et0*phi0 + et1*phi1 + et2*phi2 + et3*phi3 # det(Ft) missing
      #FI = eps0*inner(grad(et0),grad(et0)) + omega0*et0*(1.0-et0) +eps1*inner(grad(et1),grad(et1)) + omega1*et1*(1.0-et1) + eps2*inner(grad(et2),grad(et2)) + omega2*et2*(1.0-et2) + eps3*inner(grad(et3),grad(et3)) + omega3*et3*(1.0-et3)
      
      F = self.FB #+ FI
      Diss = 0.5*(nu0*(et0-et0p)*(et0-et0p) + nu1*(et1-et1p)*(et1-et1p) + nu2*(et2-et2p)*(et2-et2p) + nu3*(et3-et3p)*(et3-et3p)) / self.dt
      Pi = (F + Diss)*dx

      pen0 = conditional(lt(et0, 0.0), rho, 0.0)*et0*et0
      pen1 = conditional(lt(et1, 0.0), rho, 0.0)*et1*et1
      pen2 = conditional(lt(et2, 0.0), rho, 0.0)*et2*et2
      pen3 = conditional(lt(et3, 0.0), rho, 0.0)*et3*et3
      
      #pen = (pen0 + pen1 + pen2 + pen3)*dx
      
      self.Rc = 50.0
      rhotwo = 500.0
      x = SpatialCoordinate(self.mesh)
      self.posz = Constant(self.Rc+self.A)
      ###gap = Expression("sqrt((posx-x[0]-u[0])*(posx-x[0]-u[0])+(posy-x[1]-u[1])*(posy-x[1]-u[1])+(posz-x[2]-u[2])*(posz-x[2]-u[2])) - Rc", posx = 0.0, posy = 0.0, posz = A + Rc, degree = 2)
      ###gap = Expression("sqrt((-x[0]-u[0])*(-x[0]-u[0])+(-x[1]-u[1])*(-x[1]-u[1])+(posz-x[2]-u[2])*(posz-x[2]-u[2])) - Rc", posz = 35.0, Rc = Rc, degree = 2)
      gap = sqrt((-x[0]-u[0])*(-x[0]-u[0])+(-x[1]-u[1])*(-x[1]-u[1])+(self.posz-x[2]-u[2])*(self.posz-x[2]-u[2])) - self.Rc
      #gap = (posz-x[2]-u[2]) # plane gap
      
      penpot = conditional(lt(gap, 0.0), rhotwo, 0.0)*gap*gap
      contact = penpot*self.ds_t
      
      #Derivatives
      self.P = Pi + contact #+pen
      self.R = derivative(self.P, self.w)
      self.J = derivative(self.R, self.w)
      
      self.pvd = File("results_%s/state.pvd" % fileName)
      
      self.problem = NonlinearVariationalProblem(self.R, self.w, bcs=self.bcs, J=self.J)

      lu = {"mat_type": "baij",
            "snes_type": "newtonls",
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_max_it": 20,
            "snes_rtol": 1e-20,
            "snes_atol": 1e-8,
            "snes_linesearch_type": "l2",
            "ksp_type": "preonly",
            "ksp_monitor": None,
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
            "snes_max_it": 40,
            "snes_rtol": 1e-10,
            "snes_atol": 1e-10,
            "snes_linesearch_type": "basic",
            #"snes_linesearch_type": "bt",
            #"snes_linesearch_type": "cp",
            "snes_linesearch_maxstep": 1.0,
            "snes_linesearch_atol": 1e-20,
            "snes_linesearch_monitor": None,
            "ksp_type": "fgmres",
            "ksp_monitor": None,
            "ksp_max_it": 100,
            #"ksp_type": "cg",
            #"ksp_max_it": 600,
            "ksp_rtol": 1e-4,
            "ksp_atol": 1e-12,
            "ksp_gmres_restart": 100,
            "ksp_converged_reason": None,
            #"ksp_monitor_true_residual": None,
            "pc_type": "mg",
            "pc_mg_cycle_type": "v",
            #"pc_mg_type": "full",
            "mg_coarse_ksp_type": "preonly",
            "mg_coarse_pc_type": "lu",
            "mg_coarse_pc_factor_mat_solver_type": "mumps",
            #"mg_coarse_ksp_monitor": None,
            "mg_coarse_mat_mumps_icntl_14": 400,
            "mg_levels_ksp_type": "chebyshev",
            #"mg_levels_ksp_monitor": None,
            "mg_levels_ksp_max_it": 4,
            "mg_levels_ksp_convergence_test": "skip",
            #"mg_levels_ksp_monitor_true_residual": None,
            #"mg_levels_ksp_norm_type": "unpreconditioned",
            "mg_levels_pc_type": "pbjacobi"
      }
      
      self.sp = mg
      opts = PETSc.Options()
      opts['mat_mumps_cntl_1']= 1e-10   # relative pivoting threshold (default 0.01, 1.0 -> full pivoting, small number means less fill-in but less stable elimination, 0.0 -> no pivoting and fails if zero on diagonal)
      opts['mat_mumps_icntl_14']= 400
      opts['mat_mumps_icntl_24']= 1

      self.solver  = NonlinearVariationalSolver(self.problem, solver_parameters=self.sp)
      self.uviz = Function(self.V, name="displacement")

   def save(self):
      uw = self.w
      self.uviz.interpolate(uw)
      self.pvd.write(self.uviz, time=self.t)

   def solve(self, t_init, t_end, velocity):
      #self.its, self.ok, _ = self.solver.solve()
      #return(self.its,self.ok)

      A=self.A
      optimal_it = 12
      dt_max = 0.25 / abs(velocity)
      #self.dt.assign(0.01)

      info("dt = {}".format(float(self.dt)))
      
      self.t=t_init
      self.w0.assign(self.w)
      self.save()
      
      #tic()
      # Time-stepping
      ii = 0
     
      failed = False
     
      while self.t < t_end + dt_max - 0.0001:

            
         info(">> ii= {0:6d}   t= {1:f}   dt= {2:.2e}   id= {3:.2e}".format(ii, self.t, float(self.dt), float(A+self.Rc-self.posz)))

         t_old=self.t
         posz_old=float(self.posz)

         self.t = float(self.t + self.dt)
         if self.t > t_end:
            self.dt.assign(t_end - self.t +float(self.dt))
            self.t = t_end

         self.posz.assign(float(self.posz - velocity*self.dt))
         self.w0.assign(self.w)

         # Compute
         try:
            solvestart = time.time()
            self.solver.solve()
            solveend = time.time()
            its = self.solver.snes.getIterationNumber()
            reason = self.solver.snes.getConvergedReason()
            ok = (reason > 0)
            
         except ConvergenceError:
            info("FAILED")
            its = self.solver.snes.getIterationNumber()
            reason = self.solver.snes.getConvergedReason()
            ok = (reason > 0)
            failed = True
            #self.solver  = NonlinearVariationalSolver(self.problem, solver_parameters=self.sp)
            
         #if (ok == 0 or reason == 7):
         if (ok == 0):
            info("Step back!")

            self.t = t_old
            self.posz.assign(posz_old)

            self.dt.assign(float(self.dt)/2.0)
            info("dt = {}".format(float(self.dt)))
            self.w.assign(self.w0)
            continue

         info("Solver time taken: %s seconds" % (solveend - solvestart))

         r = Function(self.W)
         assemble(self.R, tensor=r)
         for bc in self.bcs : bc.apply(r, u=self.w)
         with r.dat.vec_ro as v: nr=v.norm()
         info("l2-norm |r|={0:.2e} L2-norm |r|={1:.2e} ".format(nr,norm(r)))
         (xu1, xu2, xu3) = split(self.w)
         info("solution |xu1|, |xu2|, |xu3| = {{{0:.2e}, {1:.2e}, {2:.2e}}},".format(norm(xu1), norm(xu2), norm(xu3)))
         (ru1, ru2, ru3) = split(r)
         info("residua |ru1|, |ru2|, |ru3| = {{{0:.2e}, {1:.2e}, {2:.2e}}},".format(norm(ru1), norm(ru2), norm(ru3)))

         # Alternative force calculation
         w_=Function(self.W)
         DirichletBC(self.W.sub(2), 1.0, self.bndry["bottom"]).apply(w_)
         force = assemble(action(self.R, w_))
         P = diff(self.FB, self.Ff)
         force2 = assemble(-P[2,2]*self.ds_b)
         force3 = assemble(P[2,2]*self.ds_t)
         
         info("indentation, force, force2, force3 = {{{0:.2e}, {1:.2e}, {2:.2e}, {3:.2e}}},".format(float(A + self.Rc - self.posz), force, force2, force3))

         info("== ii= {0:6d}   t= {1:f}   dt= {2:.2e}   id= {3:.2e}".format(ii, self.t, float(self.dt), float(A+self.Rc-self.posz)))

         if not failed:
            if its <= optimal_it:
               new_dt=min(dt_max, float(self.dt)*min(1.2,optimal_it/(its+0.001)))
            else:
               new_dt=float(self.dt)*optimal_it/its
            self.dt.assign(0.5*(float(self.dt) + new_dt))

         failed = False
         
         ii = ii + 1
         
         # Extract solutions:
         if (ii % 1 == 0):
            self.save()

         info("Solver time taken: %s seconds" % (solveend - solvestart))
         info("t = {}".format(self.t))
         info("dt = {}".format(float(self.dt)))
         info("id = {}".format(float(A+self.Rc-self.posz)))

         if self.t == t_end: break

         if float(self.dt)<1e-8: break

         info(str(datetime.datetime.now()))



