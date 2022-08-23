import numpy as np

from two_mode_solver import fd_core, util


def test_vertical_fd_bulk():
    grid = np.arange(start=-1, stop=1, step=0.01)
    quad_coeffs = np.random.uniform(-5, 5, len(grid))
    vals = np.column_stack([y/2 * grid ** 2 for y in quad_coeffs])
    vals_flat = vals.flatten()
    diff_mat = fd_core.semi_inf_second_partial_deriviative_vertical(len(grid), len(grid), grid[1] - grid[0])
    derivs = diff_mat.dot(vals_flat)
    derivs_grid = np.reshape(derivs, (len(grid), len(grid)))[2:-2, :]
    expected_results = np.tile(np.expand_dims(quad_coeffs, 0), [len(grid)-4, 1])
    np.testing.assert_allclose(derivs_grid, expected_results, rtol=1e-4)


def test_vertical_fd_boundary():
    f = np.random.uniform(-1, 1, (10, 10))
    expected_top_vals = 1/12 * (-f[2, :] + 16 * f[1, :] - 30 * f[0, :])
    expected_second_top_vals = 1 / 12 * (-f[3, :] + 16 * f[2, :] - 30 * f[1, :] + 16 * f[0, :])
    expected_bottom_vals = 1 / 12 * (-f[-3, :] + 16 * f[-2, :] - 30 * f[-1, :])
    expected_second_bottom_vals =1 / 12 * (-f[-4, :] + 16 * f[-3, :] - 30 * f[-2, :] + 16* f[-1, :])
    diff_mat = fd_core.semi_inf_second_partial_deriviative_vertical(np.shape(f)[0], np.shape(f)[1], 1)
    results = np.reshape(diff_mat.dot(f.flatten()), np.shape(f))
    np.testing.assert_allclose(results[[0,1,-2,-1], :], np.stack([expected_top_vals, expected_second_top_vals, expected_second_bottom_vals, expected_bottom_vals]))


def test_periodic_vertical_fd_boundary():
    f = np.random.uniform(-1, 1, (10, 10))
    container = np.zeros((np.shape(f)[0] + 4, np.shape(f)[1] + 4))
    container[2:-2, 2:-2] = f
    container[0,2:-2] = f[-2, :]
    container[1, 2:-2] = f[-1, :]
    container[-2, 2:-2] = f[0, :]
    container[-1, 2:-2] = f[1, :]
    diff_mat_small = fd_core.periodic_second_partial_deriviative_vertical(np.shape(f)[0], np.shape(f)[1], 0, 1)
    diff_mat_container = fd_core.semi_inf_second_partial_deriviative_vertical(np.shape(container)[0],
                                                                                np.shape(container)[1], 1)
    results_small = np.reshape(diff_mat_small.dot(f.flatten()), np.shape(f))
    results_contained = np.reshape(diff_mat_container.dot(container.flatten()), np.shape(container))[2:-2, 2:-2]
    np.testing.assert_allclose(results_small, results_contained)


def test_periodic_horizontal_fd_boundary():
    f = np.random.uniform(-1, 1, (10, 10))
    container = np.zeros((np.shape(f)[0] + 4, np.shape(f)[1] + 4))
    container[2:-2, 2:-2] = f
    container[2:-2, 0] = f[:, -2]
    container[2:-2, 1] = f[:, -1]
    container[2:-2, -2] = f[:, 0]
    container[2:-2, -1] = f[:, 1]
    diff_mat_small = fd_core.periodic_second_partial_deriviative_horizontal(np.shape(f)[0], np.shape(f)[1], 0, 1)
    diff_mat_container = fd_core.semi_inf_second_partial_deriviative_horizontal(np.shape(container)[0],
                                                                                np.shape(container)[1], 1)
    results_small = np.reshape(diff_mat_small.dot(f.flatten()), np.shape(f))
    results_contained = np.reshape(diff_mat_container.dot(container.flatten()), np.shape(container))[2:-2, 2:-2]
    np.testing.assert_allclose(results_small, results_contained)


def test_horizontal_fd_bulk():
    grid = np.arange(start=-1, stop=1, step=0.01)
    quad_coeffs = np.random.uniform(-5, 5, len(grid))
    vals = np.row_stack([y/2 * grid ** 2 for y in quad_coeffs])
    vals_flat = vals.flatten()
    diff_mat = fd_core.semi_inf_second_partial_deriviative_horizontal(len(grid), len(grid), grid[1] - grid[0])
    derivs = diff_mat.dot(vals_flat)
    derivs_grid = np.reshape(derivs, (len(grid), len(grid)))[:, 2:-2]
    expected_results = np.tile(np.expand_dims(quad_coeffs, -1), [1, len(grid)-4])
    np.testing.assert_allclose(derivs_grid, expected_results, rtol=1e-4)


def test_horizontal_fd_boundary():
    f = np.random.uniform(-1, 1, (10, 10))
    expected_top_vals = 1/12 * (-f[:, 2] + 16 * f[:, 1] - 30 * f[:, 0])
    expected_second_top_vals = 1 / 12 * (-f[:, 3] + 16 * f[:, 2] - 30 * f[:, 1] + 16 * f[:, 0])
    expected_bottom_vals = 1 / 12 * (-f[:, -3] + 16 * f[:, -2] - 30 * f[:, -1])
    expected_second_bottom_vals =1 / 12 * (-f[:, -4] + 16 * f[:, -3] - 30 * f[:, -2] + 16* f[:, -1])
    diff_mat = fd_core.semi_inf_second_partial_deriviative_horizontal(np.shape(f)[0], np.shape(f)[1], 1)
    results = np.reshape(diff_mat.dot(f.flatten()), np.shape(f))
    boundary_results = results[:, [0,1,-2,-1]]
    expected_results = np.transpose(np.stack([expected_top_vals, expected_second_top_vals, expected_second_bottom_vals, expected_bottom_vals]))
    np.testing.assert_allclose(boundary_results, expected_results)


def test_cross_fd_bulk():
    horiz_grid, vert_grid = util.generate_symmetric_grid(50, 1)
    horiz_mesh, vert_mesh = np.meshgrid(horiz_grid, vert_grid)
    quad_cnst = np.random.uniform(-1, 1)
    f = quad_cnst * horiz_mesh * vert_mesh
    diff_mat = fd_core.semi_inf_second_partial_deriviative_cross(len(horiz_grid), len(vert_grid), vert_grid[1] - vert_grid[0], horiz_grid[1] - horiz_grid[0])
    deriv = np.reshape(diff_mat.dot(f.flatten()), (len(vert_grid), len(horiz_grid)))
    np.testing.assert_allclose(deriv[2:-2, 2:-2], np.ones_like(deriv[2:-2, 2:-2]) * quad_cnst, rtol=1e-4)


def test_cross_fd_boundary():
    f = np.random.uniform(-1, 1, (10, 10))
    container = np.zeros((np.shape(f)[0]+4, np.shape(f)[1] + 4))
    container[2:-2, 2:-2] = f
    diff_mat_small = fd_core.semi_inf_second_partial_deriviative_cross(np.shape(f)[0], np.shape(f)[1], 1, 1)
    a = np.array(np.real(diff_mat_small.todense()))
    diff_mat_container = fd_core.semi_inf_second_partial_deriviative_cross(np.shape(container)[0], np.shape(container)[1], 1, 1)
    results_small = np.reshape(diff_mat_small.dot(f.flatten()), np.shape(f))
    results_contained = np.reshape(diff_mat_container.dot(container.flatten()), np.shape(container))[2:-2, 2:-2]
    np.testing.assert_allclose(results_small, results_contained)


def test_vert_periodic_cross_fd_boundary():
    f = np.random.uniform(-1, 1, (10, 10))
    container = np.zeros((np.shape(f)[0] + 4, np.shape(f)[1] + 4))
    container[2:-2, 2:-2] = f
    container[0, 2:-2] = f[-2, :]
    container[1, 2:-2] = f[-1, :]
    container[-2, 2:-2] = f[0, :]
    container[-1, 2:-2] = f[1, :]
    diff_mat_small = fd_core.periodic_vertical_second_partial_deriviative_cross(np.shape(f)[0], np.shape(f)[1], 0, 1, 1)
    diff_mat_container = fd_core.semi_inf_second_partial_deriviative_cross(np.shape(container)[0],
                                                                              np.shape(container)[1], 1, 1)
    results_small = np.reshape(diff_mat_small.dot(f.flatten()), np.shape(f))
    results_contained = np.reshape(diff_mat_container.dot(container.flatten()), np.shape(container))[2:-2, 2:-2]
    np.testing.assert_allclose(results_small, results_contained)
