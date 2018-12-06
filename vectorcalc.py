import numpy as np
from copy import deepcopy

# Calculate matrix of distance between panels
# Calculate matrix of distance between a panel and other panels normal vector
# Calculate matrix of distance between panels and arbitrary points inside the domain


def calc_r(mid_panel, normal_panel):
    # New Version, distance of a panel with respect to other panels
    r_x = np.tensordot(mid_panel[:, 0], np.ones(len(mid_panel)), axes=0);
    r_xdist = deepcopy(r_x)
    r_y = np.tensordot(mid_panel[:, 1], np.ones(len(mid_panel)), axes=0);
    r_ydist = deepcopy(r_y)
    r_xdist[:, :] = r_xdist - np.transpose(r_x)
    r_ydist[:, :] = r_ydist - np.transpose(r_y)
    r = np.sqrt(r_xdist ** 2 + r_ydist ** 2)

    # calculate magnitude r+r_n
    rn_xdist = np.tensordot(normal_panel[:, 0], np.ones(len(normal_panel)), axes=0)
    rn_ydist = np.tensordot(normal_panel[:, 1], np.ones(len(normal_panel)), axes=0)
    rn_xdist[:, :] = r_xdist + rn_xdist
    np.fill_diagonal(rn_xdist, 0)
    rn_ydist[:, :] = r_ydist + rn_ydist
    np.fill_diagonal(rn_ydist, 0)
    rplusrn = np.sqrt(rn_xdist ** 2 + rn_ydist ** 2)

    return r, rplusrn


def calc_rho(grid, normal_panel):
    # New version recreate grid coordinate as an 2-D array
    rho_xgrid = np.tensordot(grid[:, 0], np.ones(len(normal_panel)), axes=0)
    rho_ygrid = np.tensordot(grid[:, 1], np.ones(len(normal_panel)), axes=0)
    rho_xfoil = np.tensordot(np.ones(len(grid)), normal_panel[:, 0], axes=0)
    rho_yfoil = np.tensordot(np.ones(len(grid)), normal_panel[:, 1], axes=0)
    rho_x = rho_xgrid - rho_xfoil
    rho_y = rho_ygrid - rho_yfoil
    rho = np.sqrt(rho_x ** 2 + rho_y ** 2)
    rho_n = np.NAN  # Still have no ideas bout this

    return rho, rho_n
