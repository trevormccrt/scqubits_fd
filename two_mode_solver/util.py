import numpy as np


def generate_symmetric_grid(n_grid, extent):
    step = 2 * extent/n_grid
    horiz_grid = np.arange(start=-extent, stop=extent+step, step=step)
    vert_grid = np.arange(start=-extent, stop=extent+step, step=step)
    return horiz_grid, vert_grid


def generate_vertical_periodic_grid(n_grid, extent_horizontal):
    step_periodic = 2 * np.pi/n_grid
    vertical_grid = np.arange(start=-np.pi, stop=np.pi + step_periodic, step=step_periodic)
    step_horiz = 2 * extent_horizontal/n_grid
    horizontal_grid = np.arange(start=-extent_horizontal, stop=extent_horizontal+step_horiz, step=step_horiz)
    return horizontal_grid, vertical_grid