from firedrake import *
import numpy as np


def test_callbacks():
    mesh = UnitIntervalMesh(10)

    V = FunctionSpace(mesh, "CG", 1)

    u = Function(V)
    v = TestFunction(V)

    f = Function(V).assign(1)
    temp = Function(V)
    alpha = Constant(0.0)
    beta = Constant(0.0)

    F = alpha*u*v*dx - beta*f*v*dx  # we will override alpha and beta later

    def update_alpha(current_solution):
        alpha.assign(1.0)

    def update_beta(current_solution):
        with temp.dat.vec as foo:
            current_solution.copy(foo)

        # introduce current-solution-dependent behaviour
        if temp.dat.data[0] == 0.0:
            # this is reached when calculating the initial residual
            beta.assign(1.5)
        else:
            # this is reached when calculating the residual after one iteration
            beta.assign(2.0)

    problem = NonlinearVariationalProblem(F, u)

    # Perform only one iteration; expect to get u = 1.5
    params = {"snes_linesearch_type": "basic",
              "snes_max_it": 1,
              "snes_convergence_test": "skip"}

    solver = NonlinearVariationalSolver(problem,
                                        pre_jacobian_callback=update_alpha,
                                        pre_function_callback=update_beta,
                                        solver_parameters=params)
    solver.solve()

    assert np.allclose(u.dat.data, 1.5)

    # Perform two iterations; now get u = 2.0 as expected
    u.assign(0)

    new_params = {"snes_linesearch_type": "basic",
                  "snes_max_it": 2}

    solver = NonlinearVariationalSolver(problem,
                                        pre_function_callback=update_beta,
                                        solver_parameters=new_params)
    solver.solve()

    assert np.allclose(u.dat.data, 2.0)
