import numpy as np
from scipy import sparse as spm


def semi_inf_second_partial_deriviative(grid_dimension, grid_spacing):
    diags = [-1 * np.ones(grid_dimension - 2), 16 * np.ones(grid_dimension - 1), -30 * np.ones(grid_dimension),
             16 * np.ones(grid_dimension - 1), -1 * np.ones(grid_dimension - 2)]
    return 1/(12 * grid_spacing**2) * spm.diags(diags, [-2, -1, 0, 1, 2])


def periodic_second_partial_deriviative(grid_dimension, grid_spacing, gauge_constant):
    semi_inf_mat = semi_inf_second_partial_deriviative(grid_dimension, grid_spacing)
    gauge = np.exp(1j * 2 * np.pi * gauge_constant)
    new_diags = [gauge * 16 * np.ones(1), -gauge * np.ones(2),
                 -np.conjugate(gauge) * np.ones(2), np.conjugate(gauge) * 16 * np.ones(1)]
    return semi_inf_mat + 1/(12 * grid_spacing**2) * spm.diags(
        new_diags, [grid_dimension-1 ,grid_dimension-2, -(grid_dimension-2), -(grid_dimension-1)])