import numpy as np
from scipy import sparse as spm


def semi_inf_second_partial_deriviative_vertical(vertical_dimension, horizontal_dimension, grid_spacing):
    total_mat_size = vertical_dimension * horizontal_dimension
    diagonals = [-1 * np.ones(total_mat_size - 2 * horizontal_dimension),
                 16 * np.ones(total_mat_size - horizontal_dimension),
                 -30 * np.ones(total_mat_size),
                 16 * np.ones(total_mat_size - horizontal_dimension),
                 -1 * np.ones(total_mat_size - 2 * horizontal_dimension)]
    offsets = [2 * horizontal_dimension, horizontal_dimension, 0, -horizontal_dimension, -2 * horizontal_dimension]
    return 1/(12*grid_spacing**2) * spm.diags(diagonals, offsets)


def periodic_second_partial_deriviative_vertical(vertical_dimension, horizontal_dimension, gauge_constant, grid_spacing):
    semi_inf_diff_mat = semi_inf_second_partial_deriviative_vertical(vertical_dimension, horizontal_dimension, grid_spacing)
    band = np.exp(1j * gauge_constant * 2 * np.pi) * np.ones(horizontal_dimension)
    total_mat_size = semi_inf_diff_mat.shape[0]
    new_diagonals = [16 * np.conjugate(band),
                     -1 * np.conjugate(np.concatenate([band, band])),
                     -1 * np.concatenate([band, band]),
                     16 * band]
    offsets = [total_mat_size - horizontal_dimension,
               total_mat_size - 2 * horizontal_dimension,
               -(total_mat_size - 2 * horizontal_dimension),
               -(total_mat_size - horizontal_dimension)]
    bc_mat = 1/(12*grid_spacing**2) * spm.diags(new_diagonals, offsets)
    return semi_inf_diff_mat + bc_mat


def semi_inf_second_partial_deriviative_horizontal(vertical_dimension, horizontal_dimension, grid_spacing):
    total_mat_size = vertical_dimension * horizontal_dimension
    diag_first = -1 * np.ones(total_mat_size)
    diag_first[0::horizontal_dimension] = 0
    diag_first[1::horizontal_dimension] = 0
    diag_second = 16 * np.ones(total_mat_size)
    diag_second[0::horizontal_dimension] = 0
    diag_third = -30 * np.ones(total_mat_size)
    diag_fourth = 16 * np.ones(total_mat_size)
    diag_fourth[horizontal_dimension-1::horizontal_dimension] = 0
    diag_fifth = -1 * np.ones(total_mat_size)
    diag_fifth[horizontal_dimension - 1::horizontal_dimension] = 0
    diag_fifth[horizontal_dimension - 2::horizontal_dimension] = 0
    return 1/(12*grid_spacing**2) * spm.diags(
        [diag_first[2:], diag_second[1:], diag_third, diag_fourth[:-1], diag_fifth[:-2]],
        [-2, -1, 0, 1, 2])


def periodic_second_partial_deriviative_horizontal(vertical_dimension, horizontal_dimension, gauge_constant, grid_spacing):
    semi_inf_diff_mat = semi_inf_second_partial_deriviative_horizontal(vertical_dimension, horizontal_dimension, grid_spacing)
    total_mat_size = semi_inf_diff_mat.shape[0]
    diag_1 = np.zeros(total_mat_size - horizontal_dimension + 1, dtype=np.complex128)
    diag_2 = np.zeros(total_mat_size - horizontal_dimension + 2, dtype=np.complex128)
    diag_1[0::horizontal_dimension] = 16 * np.exp(-1j * gauge_constant * 2 * np.pi)
    diag_2[0::horizontal_dimension] = -1 * np.exp(-1j * gauge_constant * 2 * np.pi)
    diag_2[1::horizontal_dimension] = -1 * np.exp(-1j * gauge_constant * 2 * np.pi)
    bc_mat = spm.diags([diag_1, diag_2, np.conjugate(diag_1), np.conjugate(diag_2)],
                       [horizontal_dimension-1, horizontal_dimension-2,
                        -(horizontal_dimension-1), -(horizontal_dimension-2)])
    return semi_inf_diff_mat + 1/(12*grid_spacing**2) * bc_mat


def semi_inf_second_partial_deriviative_cross(vertical_dimension, horizontal_dimension, grid_spacing_vertical, grid_spacing_horizontal):
    total_mat_size = vertical_dimension * horizontal_dimension
    mask_1_lower = np.ones(total_mat_size)
    mask_1_lower[0::horizontal_dimension] = 0
    mask_2_lower = np.ones(total_mat_size)
    mask_2_lower[0::horizontal_dimension] = 0
    mask_2_lower[1::horizontal_dimension] = 0

    mask_1_upper = np.ones(total_mat_size)
    mask_1_upper[horizontal_dimension-1::horizontal_dimension] = 0
    mask_2_upper = np.ones(total_mat_size)
    mask_2_upper[horizontal_dimension-1::horizontal_dimension] = 0
    mask_2_upper[horizontal_dimension-2::horizontal_dimension] = 0

    diags = [
        (np.ones(total_mat_size) * mask_2_upper)[0:-(2 * horizontal_dimension + 2)],
        (-8 * np.ones(total_mat_size) * mask_1_upper)[0:-(2 * horizontal_dimension + 1)],
        (8 * np.ones(total_mat_size) * mask_1_lower)[0:-(2 * horizontal_dimension - 1)],
        (-1 * np.ones(total_mat_size) * mask_2_lower)[0:-(2 * horizontal_dimension - 2)],

        (-8 * np.ones(total_mat_size) * mask_2_upper)[0:-(horizontal_dimension + 2)],
        (64 * np.ones(total_mat_size) * mask_1_upper)[0:-(horizontal_dimension + 1)],
        (-64 * np.ones(total_mat_size) * mask_1_lower)[0:-(horizontal_dimension - 1)],
        (8 * np.ones(total_mat_size) * mask_2_lower)[0:-(horizontal_dimension - 2)],

        (-8 * np.ones(total_mat_size) * mask_2_upper)[0:-(horizontal_dimension + 2)],
        (64 * np.ones(total_mat_size) * mask_1_upper)[0:-(horizontal_dimension + 1)],
        (-64 * np.ones(total_mat_size) * mask_1_lower)[0:-(horizontal_dimension - 1)],
        (8 * np.ones(total_mat_size) * mask_2_lower)[0:-(horizontal_dimension - 2)],

        (np.ones(total_mat_size) * mask_2_upper)[0:-(2 * horizontal_dimension + 2)],
        (-8 * np.ones(total_mat_size) * mask_1_upper)[0:-(2 * horizontal_dimension + 1)],
        (8 * np.ones(total_mat_size) * mask_1_lower)[0:-(2 * horizontal_dimension - 1)],
        (-np.ones(total_mat_size) * mask_2_lower)[0:-(2 * horizontal_dimension - 2)],
    ]
    offsets = [
        2 * horizontal_dimension + 2,
        2 * horizontal_dimension + 1,
        2 * horizontal_dimension - 1,
        2 * horizontal_dimension - 2,

        horizontal_dimension + 2,
        horizontal_dimension + 1,
        horizontal_dimension - 1,
        horizontal_dimension - 2,

        -(horizontal_dimension + 2),
        -(horizontal_dimension + 1),
        -(horizontal_dimension - 1),
        -(horizontal_dimension - 2),

        -(2 * horizontal_dimension + 2),
        -(2 * horizontal_dimension + 1),
        -(2 * horizontal_dimension - 1),
        -(2 * horizontal_dimension - 2),
    ]
    return 1/(144 * grid_spacing_vertical * grid_spacing_horizontal) * spm.diags(diags, offsets)


def periodic_vertical_second_partial_deriviative_cross(vertical_dimension, horizontal_dimension, gauge_constant, grid_spacing_vertical, grid_spacing_horizontal):
    total_mat_size = vertical_dimension * horizontal_dimension
    mask_1_lower = np.ones(total_mat_size)
    mask_1_lower[0::horizontal_dimension] = 0
    mask_2_lower = np.ones(total_mat_size)
    mask_2_lower[0::horizontal_dimension] = 0
    mask_2_lower[1::horizontal_dimension] = 0

    mask_1_upper = np.ones(total_mat_size)
    mask_1_upper[horizontal_dimension - 1::horizontal_dimension] = 0
    mask_2_upper = np.ones(total_mat_size)
    mask_2_upper[horizontal_dimension - 1::horizontal_dimension] = 0
    mask_2_upper[horizontal_dimension - 2::horizontal_dimension] = 0
    gauge = np.exp(-1j * 2 * np.pi * gauge_constant)

    diags = [
        gauge * (np.ones(total_mat_size) * mask_2_upper)[-(2 * horizontal_dimension + 2):],
        gauge * (-8 * np.ones(total_mat_size) * mask_1_upper)[-(2 * horizontal_dimension + 1):],
        gauge * (8 * np.ones(total_mat_size) * mask_1_lower)[-(2 * horizontal_dimension - 1):],
        gauge * (-1 * np.ones(total_mat_size) * mask_2_lower)[-(2 * horizontal_dimension - 2):],

        gauge * (-8 * np.ones(total_mat_size) * mask_2_upper)[-(horizontal_dimension + 2):],
        gauge * (64 * np.ones(total_mat_size) * mask_1_upper)[-(horizontal_dimension + 1):],
        gauge * (-64 * np.ones(total_mat_size) * mask_1_lower)[-(horizontal_dimension - 1):],
        gauge * (8 * np.ones(total_mat_size) * mask_2_lower)[-(horizontal_dimension - 2):],

        np.conjugate(gauge) * (-8 * np.ones(total_mat_size) * mask_2_upper)[-(horizontal_dimension + 2):],
        np.conjugate(gauge) * (64 * np.ones(total_mat_size) * mask_1_upper)[-(horizontal_dimension + 1):],
        np.conjugate(gauge) * (-64 * np.ones(total_mat_size) * mask_1_lower)[-(horizontal_dimension - 1):],
        np.conjugate(gauge) * (8 * np.ones(total_mat_size) * mask_2_lower)[-(horizontal_dimension - 2):],

        np.conjugate(gauge) * (np.ones(total_mat_size) * mask_2_upper)[-(2 * horizontal_dimension + 2):],
        np.conjugate(gauge) * (-8 * np.ones(total_mat_size) * mask_1_upper)[-(2 * horizontal_dimension + 1):],
        np.conjugate(gauge) * (8 * np.ones(total_mat_size) * mask_1_lower)[-(2 * horizontal_dimension - 1):],
        np.conjugate(gauge) * (-np.ones(total_mat_size) * mask_2_lower)[-(2 * horizontal_dimension - 2):],
    ]
    offsets = [
        total_mat_size - (2 * horizontal_dimension + 2),
        total_mat_size - (2 * horizontal_dimension + 1),
        total_mat_size - (2 * horizontal_dimension - 1),
        total_mat_size - (2 * horizontal_dimension - 2),

        total_mat_size - (horizontal_dimension + 2),
        total_mat_size - (horizontal_dimension + 1),
        total_mat_size - (horizontal_dimension - 1),
        total_mat_size - (horizontal_dimension - 2),

        -(total_mat_size - (horizontal_dimension + 2)),
        -(total_mat_size - (horizontal_dimension + 1)),
        -(total_mat_size - (horizontal_dimension - 1)),
        -(total_mat_size - (horizontal_dimension - 2)),

        -(total_mat_size - (2 * horizontal_dimension + 2)),
        -(total_mat_size - (2 * horizontal_dimension + 1)),
        -(total_mat_size - (2 * horizontal_dimension - 1)),
        -(total_mat_size - (2 * horizontal_dimension - 2)),
    ]
    semi_inf_mat = semi_inf_second_partial_deriviative_cross(vertical_dimension, horizontal_dimension, grid_spacing_vertical, grid_spacing_horizontal)
    return 1/(144 * grid_spacing_vertical * grid_spacing_horizontal) * spm.diags(diags, offsets)  + semi_inf_mat
