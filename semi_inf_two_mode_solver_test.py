#import pytest
import numpy as np
from scipy.sparse import linalg as splinalg
import scqubits

import semi_inf_two_mode_solver


def test_vertical_fd():
    grid = np.arange(start=-1, stop=1, step=0.01)
    quad_coeffs = np.random.uniform(-5, 5, len(grid))
    vals = np.column_stack([y/2 * grid ** 2 for y in quad_coeffs])
    vals_flat = vals.flatten()
    diff_mat = semi_inf_two_mode_solver.second_partial_deriviative_vertical(len(grid), len(grid), grid[1] - grid[0])
    derivs = diff_mat.dot(vals_flat)
    derivs_grid = np.reshape(derivs, (len(grid), len(grid)))[2:-2, :]
    expected_results = np.tile(np.expand_dims(quad_coeffs, 0), [len(grid)-4, 1])
    np.testing.assert_allclose(derivs_grid, expected_results, rtol=1e-4)


def test_horizontal_fd():
    grid = np.arange(start=-1, stop=1, step=0.01)
    quad_coeffs = np.random.uniform(-5, 5, len(grid))
    vals = np.row_stack([y/2 * grid ** 2 for y in quad_coeffs])
    vals_flat = vals.flatten()
    diff_mat = semi_inf_two_mode_solver.second_partial_deriviative_horizontal(len(grid), len(grid), grid[1] - grid[0])
    derivs = diff_mat.dot(vals_flat)
    derivs_grid = np.reshape(derivs, (len(grid), len(grid)))[:, 2:-2]
    expected_results = np.tile(np.expand_dims(quad_coeffs, -1), [1, len(grid)-4])
    np.testing.assert_allclose(derivs_grid, expected_results, rtol=1e-4)


def _generate_grid(n_grid, extent):
    step = 2 * extent/n_grid
    horiz_grid = np.arange(start=-extent, stop=extent+step, step=step)
    vert_grid = np.arange(start=-extent, stop=extent+step, step=step)
    return horiz_grid, vert_grid


def test_uncoupled_harmonic_oscillator_eigenvalues():
    horiz_grid, vert_grid = _generate_grid(100, 6 * np.pi)
    ec = np.random.uniform(1, 3, 2)
    el = np.random.uniform(1, 3, 2)
    ham = semi_inf_two_mode_solver.generate_hamiltonian(horiz_grid, vert_grid, ec, np.diag(el),0, [0,0,0], [0,0,0])
    vals, _ = splinalg.eigsh(ham, 4, which="SA")
    omegas = sorted([vals[1] - vals[0], vals[2] - vals[0]])
    omega_act = sorted(np.sqrt(4 * ec * el))
    np.testing.assert_allclose(omegas, omega_act, rtol=1e-2)


def test_uncoupled_fluxonium():
    np.random.seed(1000)
    n_grid = 100
    ec_vals = np.random.uniform(1,3, 2)
    el_vals = np.random.uniform(0.4, 1, 2)
    ej_vals = np.random.uniform(7, 10, 2)
    fluxes = np.random.uniform(0, 0.5, 2)
    scqubits_vals = []
    for ec, el, ej, flux in zip(ec_vals, el_vals, ej_vals, fluxes):
        f = scqubits.Fluxonium(ej, ec, el, flux, cutoff=500)
        s_vals = f.eigenvals(2)
        scqubits_vals.append(s_vals[1]- s_vals[0])
    horiz_grid, vert_grid = _generate_grid(n_grid, 8 * np.pi)
    ham = semi_inf_two_mode_solver.generate_hamiltonian(
        horiz_grid, vert_grid, 4 * ec_vals, np.diag(el_vals)/2, 0,
        np.concatenate([ej_vals, [0]]), 2*np.pi * np.concatenate([fluxes, [0]]))
    vals, _ = splinalg.eigsh(ham, 4, which="SA")
    scqubits_vals = sorted(scqubits_vals)
    freqs = sorted([vals[1] - vals[0], vals[3] - vals[0]])
    np.testing.assert_allclose(scqubits_vals, freqs, rtol=1e-2)


def test_double_fluxonium():
    ec = 4 * 3.4
    el = 1.2
    ej = 9.4
    alpha = 0.006
    el_mat = np.array([[1/3, 1/6], [1/6, 1/3]]) * el
    ej_vals = [ej, ej, 0]
    ext_fluxes = [np.pi * (1-alpha/2), np.pi * (1+alpha/2), 0]
    horiz_grid, vert_grid = _generate_grid(200, 6 * np.pi)
    ham = semi_inf_two_mode_solver.generate_hamiltonian(horiz_grid, vert_grid, [ec, ec], el_mat, 0, ej_vals, ext_fluxes)
    vals, _ = splinalg.eigsh(ham, 4, which="SA")
    freqs = np.diff(sorted(vals))
    np.testing.assert_allclose(freqs[0], 0.105, rtol=1e-2)
