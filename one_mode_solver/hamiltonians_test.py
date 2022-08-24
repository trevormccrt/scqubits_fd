import numpy as np
import scqubits
from scipy.sparse import linalg as splinalg

from one_mode_solver import hamiltonians


def test_transmon():
    ec = np.random.uniform(1,5)
    ej = np.random.uniform(1,15)
    ng = np.random.uniform(0,2)
    n_grid = 200
    step = 2 * np.pi/n_grid
    grid = np.arange(start=-np.pi, stop=np.pi + step, step=step)
    ham = hamiltonians.periodic_hamiltonian(grid, ec, ng, ej, 0)
    vals, _ = splinalg.eigsh(ham, 2, which="SA")
    trans_sc = scqubits.Transmon(ej, ec/4, ng, ncut=n_grid)
    sc_vals = trans_sc.eigenvals(2)
    np.testing.assert_allclose(vals[1] - vals[0], sc_vals[1] - sc_vals[0], rtol=1e-2)


def test_fluxonium():
    ec = np.random.uniform(1, 5)
    ej = np.random.uniform(1, 15)
    el = np.random.uniform(1,3)
    flux = np.random.uniform(0,3)
    extent = 6 * np.pi
    n_grid = 200
    step = 2 * extent/n_grid
    grid = np.arange(start=-extent, stop=extent, step=step)
    ham = hamiltonians.semi_inf_hamiltonian(grid, ec, el, ej, 2 * np.pi * flux)
    vals, _ = splinalg.eigsh(ham, 2, which="SA")
    flux_sc = scqubits.Fluxonium(ej, ec/4, 2 * el, flux, n_grid)
    sc_vals = flux_sc.eigenvals(2)
    np.testing.assert_allclose(vals[1] - vals[0], sc_vals[1] - sc_vals[0], rtol=1e-2)
