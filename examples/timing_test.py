import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg as splinalg
import time

from two_mode_solver import util, periodic_semi_inf

# zero-pi like hamiltonian
ej = 0.3
el = 7e-1
ec_1 = 1
ec_2 = 2
ec_3 = 0.5
ng = 0.3
flux = 0.4

grid_sizes = [10, 15, 30, 50, 75, 100, 150, 200, 250]
eig_times = []
eig_freqs = []

for grid_size in grid_sizes:
    horiz_grid, vert_grid = util.generate_vertical_periodic_grid(grid_size, 6 * np.pi)
    ham = periodic_semi_inf.generate_hamiltonian(
        horiz_grid, vert_grid, np.array([[ec_1, ec_3], [ec_3, ec_2]]),
        ng, el/4, [ej, 0, ej], [-1 * 2 * np.pi * flux/2, 0, 2 * np.pi * flux/2])
    sh_start_time = time.time()
    vals, _ = splinalg.eigsh(ham, 3, which="SA")
    sh_end_time = time.time()
    sorted_vals = np.sort(vals)
    eig_freqs.append(sorted_vals[1:] - sorted_vals[0])
    eig_times.append(sh_end_time - sh_start_time)

freq_delta = [(x - eig_freqs[-1])/eig_freqs[-1] * 100 for x in eig_freqs[:-1]]
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
axs[0].plot(grid_sizes, eig_times)
axs[0].set_ylabel("Solution Time (s)")
axs[0].set_yscale("log")

axs[1].plot(grid_sizes[:-1], np.abs(freq_delta))
axs[1].set_xlabel("Grid Side Length (Symmetric)")
axs[1].set_ylabel("Frequency Convergence (%)")
axs[1].set_yscale("log")
fig.suptitle("Periodic-Confined 2D Problem")
plt.show()
