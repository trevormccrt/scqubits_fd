import numpy as np
from scipy import sparse as spm

from two_mode_solver import fd_core


def _rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def linear_potential(horizontal_vals, vertical_vals, el_mat, theta):
    horiz_grid = np.tile(np.expand_dims(horizontal_vals, 0), [len(vertical_vals), 1])
    vert_grid = np.tile(np.expand_dims(vertical_vals, -1), [1, len(horizontal_vals)])
    stacked_grid = np.stack([vert_grid, horiz_grid])
    rotation_matrix = _rotation_matrix(theta)
    return np.einsum("iab, ij, jab -> ab",
                                  stacked_grid,
                                  np.matmul(np.transpose(rotation_matrix), np.matmul(el_mat, rotation_matrix)),
                                  stacked_grid)


def cosine_potential(horizontal_vals, vertical_vals, ej_vals, external_fluxes, theta):
    horiz_grid = np.tile(np.expand_dims(horizontal_vals, 0), [len(vertical_vals), 1])
    vert_grid = np.tile(np.expand_dims(vertical_vals, -1), [1, len(horizontal_vals)])
    stacked_grid = np.stack([vert_grid, horiz_grid])
    rotation_matrix = _rotation_matrix(theta)
    shifted_grid = np.einsum("ij, jab -> iab", rotation_matrix, stacked_grid)
    return ej_vals[0]*(1-np.cos(shifted_grid[0, :, :] - external_fluxes[0])) +\
           ej_vals[1]*(1-np.cos(shifted_grid[1, :, :] - external_fluxes[1])) + \
           ej_vals[2]*(1-np.cos(shifted_grid[0, :, :] + shifted_grid[1, :, :] - external_fluxes[2]))


def generate_hamiltonian(horizontal_vals, vertical_vals, ec_vals, el_mat, theta, ej_vals, external_fluxes):
    horizontal_dimension = len(horizontal_vals)
    vertical_dimension = len(vertical_vals)
    ham = -ec_vals[1] * fd_core.semi_inf_second_partial_deriviative_horizontal(vertical_dimension, horizontal_dimension, horizontal_vals[1] - horizontal_vals[0]) + \
          -ec_vals[0] * fd_core.semi_inf_second_partial_deriviative_vertical(vertical_dimension, horizontal_dimension, vertical_vals[1] - vertical_vals[0]) + \
          spm.diags([linear_potential(horizontal_vals, vertical_vals, el_mat, theta).flatten()], [0]) + \
          spm.diags([cosine_potential(horizontal_vals, vertical_vals, ej_vals, external_fluxes, 0).flatten()], [0])
    return ham
