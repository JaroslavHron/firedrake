import ufl
from itertools import chain
from contextlib import ExitStack

from firedrake import dmhooks
from firedrake import slate
from firedrake import solving_utils
from firedrake import ufl_expr
from firedrake import utils
from firedrake.petsc import PETSc, OptionsManager

__all__ = ["VariationalProblem",
           "LinearVariationalProblem",
           "LinearVariationalSolver",
           "NonlinearVariationalProblem",
           "NonlinearVariationalSolver"]


class VariationalProblem(object):
    r"""Mixed variational problem defined by eq."""
    # It is convenient to define ancestor of LinearVariationalProblem
    # and NonlinearVariationalProblem as, if FormBCs exist, linear and
    # nonlinear forms can coexist, in which case it is cleaner to
    # deal with all cases in a monolithic framework.

    def __init__(self, eq, u, bcs=None, J=None, Jp=None,
                 form_compiler_parameters=None,
                 constant_jacobian=False):
        r"""
        :param eq: the equation form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param J: the Jacobian J = dF/du (optional)
        :param Jp: a form used for preconditioning the linear system,
                 optional, if not supplied then the Jacobian itself
                 will be used.
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        """
        from firedrake import solving
        from firedrake import function
        from firedrake.bcs import FormBC

        self.bcs = solving._extract_bcs(bcs)

        # Store solution Function
        self.u = u
        # Argument checking
        if not isinstance(self.u, function.Function):
            raise TypeError("Provided solution is a '%s', not a Function" % type(self.u).__name__)

        # This has to stay True if LinearVariationalProblem is being solved.
        self.is_linear = True

        # For domain equation form and boundary equation forms (FormBC.eq)
        # Store input UFL residual forms
        # Store input UFL Jacobian/preconditionar forms
        # Use the user-provided Jacobian. If none is provided, derive
        # the Jacobian from the residual.

        # Domain form
        # linear
        if isinstance(eq.lhs, ufl.Form) and isinstance(eq.rhs, ufl.Form):
            self.J = eq.lhs
            self.Jp = Jp
            if eq.rhs is 0:
                self.F = ufl_expr.action(self.J, u)
            else:
                if not isinstance(eq.rhs, (ufl.Form, slate.slate.TensorBase)):
                    raise TypeError("Provided RHS is a '%s', not a Form or Slate Tensor" % type(eq.rhs).__name__)
                if len(eq.rhs.arguments()) != 1:
                    raise ValueError("Provided RHS is not a linear form")
                self.F = ufl_expr.action(self.J, u) - eq.rhs
        # nonlinear
        else:
            if eq.rhs != 0:
                raise TypeError("r.h.s. of nonlinear form has to be 0")
            if not isinstance(eq.lhs, (ufl.Form, slate.slate.TensorBase)):
                raise TypeError("Provided residual is a '%s', not a Form or Slate Tensor" % type(eq.lhs).__name__)
            if len(eq.lhs.arguments()) != 1:
                raise ValueError("Provided residual is not a linear form")
            self.F = eq.lhs
            self.J = J or ufl_expr.derivative(self.F, u)
            self.Jp = Jp
            if not isinstance(self.J, (ufl.Form, slate.slate.TensorBase)):
                raise TypeError("Provided Jacobian is a '%s', not a Form or Slate Tensor" % type(self.J).__name__)
            if len(self.J.arguments()) != 2:
                raise ValueError("Provided Jacobian is not a bilinear form")
            if self.Jp is not None and not isinstance(self.Jp, (ufl.Form, slate.slate.TensorBase)):
                raise TypeError("Provided preconditioner is a '%s', not a Form or Slate Tensor" % type(self.Jp).__name__)
            if self.Jp is not None and len(self.Jp.arguments()) != 2:
                raise ValueError("Provided preconditioner is not a bilinear form")
            self.is_linear = False

        # Boundary (FormBC)
        if self.bcs is not None:
            for bc in self.bcs:
                if isinstance(bc, FormBC):
                    # linear
                    if isinstance(bc.formeq.lhs, ufl.Form) and isinstance(bc.formeq.rhs, ufl.Form):
                        bc.J = bc.formeq.lhs
                        if bc.formeq.rhs is 0:
                            bc.F = ufl_expr.action(bc.J, u)
                        else:
                            if not isinstance(bc.formeq.rhs, (ufl.Form, slate.slate.TensorBase)):
                                raise TypeError("Provided BC RHS is a '%s', not a Form or Slate Tensor" % type(bc.formeq.rhs).__name__)
                            if len(bc.formeq.rhs.arguments()) != 1:
                                raise ValueError("Provided BC RHS is not a linear form")
                            bc.F = ufl_expr.action(bc.J, u) - bc.formeq.rhs
                    # nonlinear
                    else:
                        if bc.formeq.rhs != 0:
                            raise TypeError("r.h.s. of nonlinear form has to be 0")
                        if not isinstance(bc.formeq.lhs, (ufl.Form, slate.slate.TensorBase)):
                            raise TypeError("Provided BC residual is a '%s', not a Form or Slate Tensor" % type(bc.formeq.lhs).__name__)
                        if len(bc.formeq.lhs.arguments()) != 1:
                            raise ValueError("Provided BC residual is not a linear form")
                        bc.F = bc.formeq.lhs
                        bc.J = bc.J or ufl_expr.derivative(bc.F, u)
                        if not isinstance(bc.J, (ufl.Form, slate.slate.TensorBase)):
                            raise TypeError("Provided BC Jacobian is a '%s', not a Form or Slate Tensor" % type(bc.J).__name__)
                        if len(bc.J.arguments()) != 2:
                            raise ValueError("Provided BC Jacobian is not a bilinear form")
                        if bc.Jp is not None and not isinstance(bc.Jp, (ufl.Form, slate.slate.TensorBase)):
                            raise TypeError("Provided BC preconditioner is a '%s', not a Form or Slate Tensor" % type(bc.Jp).__name__)
                        if bc.Jp is not None and len(bc.Jp.arguments()) != 2:
                            raise ValueError("Provided BC preconditioner is not a bilinear form")
                        self.is_linear = False

        if not self.is_linear:
            constant_jacobian = False

        # Store form compiler parameters
        self.form_compiler_parameters = form_compiler_parameters
        self._constant_jacobian = constant_jacobian

    @utils.cached_property
    def dm(self):
        return self.u.function_space().dm


class NonlinearVariationalProblem(VariationalProblem):
    r"""Nonlinear variational problem F(u; v) = 0."""

    def __init__(self, F, u, bcs=None, J=None,
                 Jp=None,
                 form_compiler_parameters=None):
        r"""
        :param F: the nonlinear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param J: the Jacobian J = dF/du (optional)
        :param Jp: a form used for preconditioning the linear system,
                 optional, if not supplied then the Jacobian itself
                 will be used.
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        """

        super(NonlinearVariationalProblem, self).__init__(F == 0, u, bcs=bcs, J=J, Jp=Jp,
                                                          form_compiler_parameters=form_compiler_parameters)


class NonlinearVariationalSolver(OptionsManager):
    r"""Solves a :class:`NonlinearVariationalProblem`."""

    def __init__(self, problem, **kwargs):
        r"""
        :arg problem: A :class:`NonlinearVariationalProblem` to solve.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.
        :kwarg transpose_nullspace: as for the nullspace, but used to
               make the right hand side consistent.
        :kwarg near_nullspace: as for the nullspace, but used to
               specify the near nullspace (for multigrid solvers).
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
               This should be a dict mapping PETSc options to values.
        :kwarg appctx: A dictionary containing application context that
               is passed to the preconditioner if matrix-free.
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.
        :kwarg pre_jacobian_callback: A user-defined function that will
               be called immediately before Jacobian assembly. This can
               be used, for example, to update a coefficient function
               that has a complicated dependence on the unknown solution.
        :kwarg pre_function_callback: As above, but called immediately
               before residual assembly

        Example usage of the ``solver_parameters`` option: to set the
        nonlinear solver type to just use a linear solver, use

        .. code-block:: python

            {'snes_type': 'ksponly'}

        PETSc flag options should be specified with `bool` values.
        For example:

        .. code-block:: python

            {'snes_monitor': True}

        To use the ``pre_jacobian_callback`` or ``pre_function_callback``
        functionality, the user-defined function must accept the current
        solution as a petsc4py Vec. Example usage is given below:

        .. code-block:: python

            def update_diffusivity(current_solution):
                with cursol.dat.vec_wo as v:
                    current_solution.copy(v)
                solve(trial*test*dx == dot(grad(cursol), grad(test))*dx, diffusivity)

            solver = NonlinearVariationalSolver(problem,
                                                pre_jacobian_callback=update_diffusivity)

        """
        assert isinstance(problem, VariationalProblem)

        parameters = kwargs.get("solver_parameters")
        if "parameters" in kwargs:
            raise TypeError("Use solver_parameters, not parameters")
        nullspace = kwargs.get("nullspace")
        nullspace_T = kwargs.get("transpose_nullspace")
        near_nullspace = kwargs.get("near_nullspace")
        options_prefix = kwargs.get("options_prefix")
        pre_j_callback = kwargs.get("pre_jacobian_callback")
        pre_f_callback = kwargs.get("pre_function_callback")

        super(NonlinearVariationalSolver, self).__init__(parameters, options_prefix)

        # Allow anything, interpret "matfree" as matrix_free.
        mat_type = self.parameters.get("mat_type")
        pmat_type = self.parameters.get("pmat_type")
        matfree = mat_type == "matfree"
        pmatfree = pmat_type == "matfree"

        appctx = kwargs.get("appctx")

        ctx = solving_utils._SNESContext(problem,
                                         mat_type=mat_type,
                                         pmat_type=pmat_type,
                                         appctx=appctx,
                                         pre_jacobian_callback=pre_j_callback,
                                         pre_function_callback=pre_f_callback,
                                         options_prefix=self.options_prefix)

        # No preconditioner by default for matrix-free
        if (problem.Jp is not None and pmatfree) or matfree:
            self.set_default_parameter("pc_type", "none")
        elif ctx.is_mixed:
            # Mixed problem, use jacobi pc if user has not supplied
            # one.
            self.set_default_parameter("pc_type", "jacobi")

        self.snes = PETSc.SNES().create(comm=problem.dm.comm)

        self._problem = problem

        self._ctx = ctx
        self._work = problem.u.dof_dset.layout_vec.duplicate()
        self.snes.setDM(problem.dm)

        ctx.set_function(self.snes)
        ctx.set_jacobian(self.snes)
        ctx.set_nullspace(nullspace, problem.J.arguments()[0].function_space()._ises,
                          transpose=False, near=False)
        ctx.set_nullspace(nullspace_T, problem.J.arguments()[1].function_space()._ises,
                          transpose=True, near=False)
        ctx.set_nullspace(near_nullspace, problem.J.arguments()[0].function_space()._ises,
                          transpose=False, near=True)

        # Set from options now, so that people who want to noodle with
        # the snes object directly (mostly Patrick), can.  We need the
        # DM with an app context in place so that if the DM is active
        # on a subKSP the context is available.
        dm = self.snes.getDM()
        with dmhooks.appctx(dm, self._ctx):
            self.set_from_options(self.snes)

        # Used for custom grid transfer.
        self._transfer_operators = ()
        self._setup = False

    def set_transfer_operators(self, *contextmanagers):
        r"""Set context managers which manages which grid transfer operators should be used.

        :arg contextmanagers: instances of :class:`~.dmhooks.transfer_operators`.
        :raises RuntimeError: if called after calling solve.
        """
        if self._setup:
            raise RuntimeError("Cannot set transfer operators after solve")
        self._transfer_operators = tuple(contextmanagers)

    def solve(self, bounds=None):
        r"""Solve the variational problem.

        :arg bounds: Optional bounds on the solution (lower, upper).
            ``lower`` and ``upper`` must both be
            :class:`~.Function`\s. or :class:`~.Vector`\s.

        .. note::

           If bounds are provided the ``snes_type`` must be set to
           ``vinewtonssls`` or ``vinewtonrsls``.
        """
        # Make sure appcontext is attached to the DM before we solve.
        dm = self.snes.getDM()
        # Apply the boundary conditions to the initial guess.
        from firedrake.bcs import DirichletBC
        for bc in self._problem.bcs:
            if isinstance(bc, DirichletBC):
                bc.apply(self._problem.u)

        if bounds is not None:
            lower, upper = bounds
            with lower.dat.vec_ro as lb, upper.dat.vec_ro as ub:
                self.snes.setVariableBounds(lb, ub)
        work = self._work
        with self._problem.u.dat.vec as u:
            u.copy(work)
            with ExitStack() as stack:
                # Ensure options database has full set of options (so monitors
                # work right)
                for ctx in chain((self.inserted_options(), dmhooks.appctx(dm, self._ctx)),
                                 self._transfer_operators):
                    stack.enter_context(ctx)
                self.snes.solve(None, work)
            work.copy(u)
        self._setup = True
        solving_utils.check_snes_convergence(self.snes)


class LinearVariationalProblem(VariationalProblem):
    r"""Linear variational problem a(u, v) = L(v)."""

    def __init__(self, a, L, u, bcs=None, aP=None,
                 form_compiler_parameters=None,
                 constant_jacobian=True):
        r"""
        :param a: the bilinear form
        :param L: the linear form
        :param u: the :class:`.Function` to solve for
        :param bcs: the boundary conditions (optional)
        :param aP: an optional operator to assemble to precondition
                 the system (if not provided a preconditioner may be
                 computed from ``a``)
        :param dict form_compiler_parameters: parameters to pass to the form
            compiler (optional)
        :param constant_jacobian: (optional) flag indicating that the
                 Jacobian is constant (i.e. does not depend on
                 varying fields).  If your Jacobian can change, set
                 this flag to ``False``.
        """

        super(LinearVariationalProblem, self).__init__(a == L, u, bcs=bcs, J=None, Jp=aP,
                                                       form_compiler_parameters=form_compiler_parameters,
                                                       constant_jacobian=constant_jacobian)
        if not self.is_linear:
            raise TypeError("Not all domain/boundary forms provided are of linear form")


class LinearVariationalSolver(NonlinearVariationalSolver):
    r"""Solves a :class:`LinearVariationalProblem`."""

    def __init__(self, *args, **kwargs):
        r"""
        :arg problem: A :class:`LinearVariationalProblem` to solve.
        :kwarg solver_parameters: Solver parameters to pass to PETSc.
            This should be a dict mapping PETSc options to values.
        :kwarg nullspace: an optional :class:`.VectorSpaceBasis` (or
               :class:`.MixedVectorSpaceBasis`) spanning the null
               space of the operator.
        :kwarg transpose_nullspace: as for the nullspace, but used to
               make the right hand side consistent.
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.
        :kwarg appctx: A dictionary containing application context that
               is passed to the preconditioner if matrix-free.
        """
        parameters = {}
        parameters.update(kwargs.get("solver_parameters", {}))
        parameters.setdefault('snes_type', 'ksponly')
        parameters.setdefault('ksp_rtol', 1.0e-7)
        kwargs["solver_parameters"] = parameters
        super(LinearVariationalSolver, self).__init__(*args, **kwargs)

    def invalidate_jacobian(self):
        r"""
        Forces the matrix to be reassembled next time it is required.
        """
        self._ctx._jacobian_assembled = False
