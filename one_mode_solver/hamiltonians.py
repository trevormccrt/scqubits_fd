import numpy as np
from scipy import sparse as spm

from one_mode_solver import fd_core


def linear_potential(grid, el):
    return el * grid**2


def cosine_potential(grid, ej, flux):
    return ej * (1 - np.cos(grid - flux))


def semi_inf_hamiltonian(grid, ec, el, ej, flux):
    return -ec * fd_core.semi_inf_second_partial_deriviative(len(grid), grid[1] - grid[0]) +\
           spm.diags([linear_potential(grid, el)], [0]) + spm.diags([cosine_potential(grid, ej, flux)], [0])


def periodic_hamiltonian(grid, ec, ng, ej, flux):
    return -ec * fd_core.periodic_second_partial_deriviative(len(grid), grid[1] - grid[0], ng) +\
           spm.diags([cosine_potential(grid, ej, flux)], [0])