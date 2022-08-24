import numpy as np
from scipy.sparse import linalg as splinalg
import time
import matplotlib.pyplot as plt

from two_mode_solver import periodic_semi_inf, util

# zero-pi like hamiltonian
ej = 0.3
el = 7e-1
ec_1 = 1
ec_2 = 2
ec_3 = 0.5
ng = 0.3
flux = 0.4


# solve EVP at different resolutions using different methods and time it
grid_sizes = [20, 50, 100, 150, 200, 250]# 200, 250, 300]
time_sa = []
time_sm = []
time_la = []
time_lm = []
all_vals_sa = []
all_vals_sm = []
all_vals_la = []
all_vals_lm = []
for grid_size in grid_sizes:
    horiz_grid, vert_grid = util.generate_vertical_periodic_grid(grid_size, 6 * np.pi)
    ham = periodic_semi_inf.generate_hamiltonian(
        horiz_grid, vert_grid, np.array([[ec_1, ec_3], [ec_3, ec_2]]),
        ng, el / 4, [ej, 0, ej], [-1 * 2 * np.pi * flux / 2, 0, 2 * np.pi * flux / 2])
    t1_sa = time.time()
    vals_sa, _ = splinalg.eigsh(ham, 2, which="SA")
    t2_sa = time.time()
    time_sa.append(t2_sa - t1_sa)
    all_vals_sa.append(vals_sa[1] - vals_sa[0])
    t1_sm = time.time()
    vals_sm, _ = splinalg.eigsh(ham, 2, which="SM")
    t2_sm = time.time()
    time_sm.append(t2_sm - t1_sm)
    all_vals_sm.append(vals_sm[1] - vals_sm[0])
    t1_lm = time.time()
    vals_lm, _ = splinalg.eigsh(ham, 2, sigma=0, which="LM")
    t2_lm = time.time()
    time_lm.append(t2_lm - t1_lm)
    all_vals_lm.append(vals_lm[1] - vals_lm[0])
    t1_la = time.time()
    vals_la, _ = splinalg.eigsh(ham, 2, sigma=0, which="LA")
    t2_la = time.time()
    time_la.append(t2_la - t1_la)
    all_vals_la.append(vals_la[1] - vals_la[0])


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
axs[0].plot(grid_sizes, time_sa, label="SA")
axs[0].plot(grid_sizes, time_sm, label="SM")
axs[0].plot(grid_sizes, time_la, label="LA")
axs[0].plot(grid_sizes, time_lm, label="LM")
axs[0].legend()
axs[0].set_ylabel("Solution Time (s)")
axs[0].set_yscale("log")

axs[1].plot(grid_sizes, all_vals_sa)
axs[1].plot(grid_sizes, all_vals_sm)
axs[1].plot(grid_sizes, all_vals_la)
axs[1].plot(grid_sizes, all_vals_lm)
axs[1].set_ylabel("f10")

axs[1].set_xlabel("Grid Side Length")
plt.show()
