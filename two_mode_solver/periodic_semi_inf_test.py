import numpy as np
from scipy.sparse import linalg as splinalg
import scqubits

from two_mode_solver import periodic_semi_inf, util


def test_uncoupled_oscillator_transmon():
    horiz_grid, vert_grid = util.generate_vertical_periodic_grid(100, 6 * np.pi)
    ng = np.random.uniform(0,2)
    ec_osc = np.random.uniform(1,3)
    el_osc = np.random.uniform(4, 7)
    ec_trans = ec_osc + np.random.uniform(-0.3 * ec_osc, 0.3 * ec_osc)
    ej_trans = el_osc + np.random.uniform(-0.3 * el_osc, 0.3 * el_osc)
    ham = periodic_semi_inf.generate_hamiltonian(horiz_grid, vert_grid, np.array([[ec_trans, 0], [0, ec_osc]]), ng, el_osc, [ej_trans, 0, 0], [0, 0, 0])
    vals, _ = splinalg.eigsh(ham, 5, which="SA")
    vals = sorted(vals)
    osc_freq = np.sqrt(4 * ec_osc * el_osc)
    transmon = scqubits.Transmon(ej_trans, ec_trans/4, ng, 200)
    vals_sc = transmon.eigenvals(2)
    freqs_mine = sorted([vals[1] - vals[0], vals[2] - vals[0]])
    freqs_expected = sorted([osc_freq, vals_sc[1] - vals_sc[0]])
    np.testing.assert_allclose(freqs_mine, freqs_expected, rtol=1e-2)


def test_zero_pi():
    ej = 0.3
    el = 7e-1
    ecj = 1
    ecsig = 3e-1
    ng = 0.3
    flux = 0.4
    horiz_grid, vert_grid = util.generate_vertical_periodic_grid(200, 6 * np.pi)
    ham = periodic_semi_inf.generate_hamiltonian(
        horiz_grid, vert_grid, np.array([[2 * (ecj + ecsig), -4 * ecj],
                                        [-4 * ecj, 8 * ecj]]),
        ng, el/4, [ej, 0, ej], [-1 * 2 * np.pi * flux/2, 0, 2 * np.pi * flux/2])
    vals, _ = splinalg.eigsh(ham, 3, which="SA")
    vals = np.array(sorted(vals))
    freqs = vals[1:] - vals[0]
    phi_grid = scqubits.Grid1d(-6 * np.pi, 6 * np.pi, 200)
    zero_pi_scq = scqubits.ZeroPi(EJ=ej, EL=el, ECJ=ecj, EC=None, ng=ng, flux=flux,
                                  grid=phi_grid, ncut=30, ECS=ecsig, truncated_dim=200)
    sc_vals = zero_pi_scq.eigenvals(3)
    sc_vals = np.array(sorted(sc_vals))
    sc_freqs = sc_vals[1:] - sc_vals[0]
    np.testing.assert_allclose(freqs, sc_freqs, rtol=1e-2)
