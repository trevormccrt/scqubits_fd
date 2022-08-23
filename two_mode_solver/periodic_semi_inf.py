import numpy as np
from scipy import sparse as spm

from two_mode_solver import semi_inf, fd_core


def linear_potential(vertical_dim, horizontal_vals, el):
    horiz_grid = np.tile(np.expand_dims(horizontal_vals, 0), [vertical_dim, 1])
    return el * horiz_grid**2


def cosine_potential(horizontal_vals, vertical_vals, ej_vals, external_fluxes):
    return semi_inf.cosine_potential(horizontal_vals, vertical_vals, ej_vals, external_fluxes, theta=0)


def generate_hamiltonian(horizontal_vals, vertical_vals, ec_mat, ng, el,  ej_vals, external_fluxes):
    horizontal_dimension = len(horizontal_vals)
    vertical_dimension = len(vertical_vals)
    grid_spacing_horizontal = horizontal_vals[1] - horizontal_vals[0]
    grid_spacing_vertical = vertical_vals[1] - vertical_vals[0]
    return -ec_mat[0,0] * fd_core.periodic_second_partial_deriviative_vertical(
        vertical_dimension, horizontal_dimension, ng, grid_spacing_vertical) - \
        ec_mat[1, 1] * fd_core.semi_inf_second_partial_deriviative_horizontal(
        vertical_dimension, horizontal_dimension, grid_spacing_horizontal) - \
        2 * ec_mat[0, 1] * fd_core.periodic_vertical_second_partial_deriviative_cross(
        vertical_dimension, horizontal_dimension, ng, grid_spacing_vertical, grid_spacing_horizontal) + \
        spm.diags([linear_potential(vertical_dimension, horizontal_vals, el).flatten()], [0]) + \
        spm.diags([cosine_potential(horizontal_vals, vertical_vals, ej_vals, external_fluxes).flatten()], [0])


