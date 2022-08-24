import numpy as np

from one_mode_solver import fd_core


def test_semi_inf_second_deriv():
    grid = np.arange(start=-1, stop=1, step=0.01)
    quad_coeff = np.random.uniform(-5,5)
    f = quad_coeff/2 * grid**2
    deriv_mat = fd_core.semi_inf_second_partial_deriviative(len(grid), grid[1] - grid[0])
    result = deriv_mat.dot(f)
    np.testing.assert_allclose(result[2:-2], np.ones_like(result[2:-2]) * quad_coeff)


def test_semi_inf_second_deriv_boundaries():
    f = np.random.uniform(-1, 1, 20)
    container = np.zeros(len(f) + 4)
    container[2:-2] = f
    deriv_mat_small = fd_core.semi_inf_second_partial_deriviative(len(f), 1)
    result_small = deriv_mat_small.dot(f)
    deriv_mat_large = fd_core.semi_inf_second_partial_deriviative(len(container), 1)
    result_large = deriv_mat_large.dot(container)
    np.testing.assert_allclose(result_small, result_large[2:-2])


def test_periodic_second_deriv():
    f = np.random.uniform(-1, 1, 20)
    container = np.zeros(len(f) + 4)
    container[2:-2] = f
    container[0] = f[-2]
    container[1] = f[-1]
    container[-2] = f[0]
    container[-1] = f[1]
    deriv_mat_small = fd_core.periodic_second_partial_deriviative(len(f), 1, 0)
    result_small = deriv_mat_small.dot(f)
    deriv_mat_large = fd_core.semi_inf_second_partial_deriviative(len(container), 1)
    result_large = deriv_mat_large.dot(container)
    np.testing.assert_allclose(result_small, result_large[2:-2])
