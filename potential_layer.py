# Calculate single and double potential layer
# Solve h using integral phi and g as the boundary condition
# Calculate potential using integral of h and phi

import numpy as np
import calc_phi as cp
import vectorcalc


# Potential solver
def pot(mid_panel, normal_panel, length_panel, g, u_inf, layer, grid):

    # ============================= VARIABLES INITIALIZATION =============================
    # OLD VERSION
    # r_vec = np.linalg.norm(mid_panel, axis=1)
    # r_norm = np.linalg.norm(normal_panel, axis=1)
    # x_vec = np.linalg.norm(grid, axis=1)
    #
    # # recreate r, r_n and ds as an 2-D array
    # r = np.tensordot(r_vec, np.ones(len(r_vec)), axes=0)
    # r[:, :] = r - np.transpose(r)
    # r_n = np.tensordot(np.ones(len(r_norm)), r_norm, axes=0)

    # #Old version recreate grid coordinate as an 2-D array
    # x = np.tensordot(x_vec, np.ones(len(r_vec)), axes=0)
    # rp = np.tensordot(np.ones(len(x_vec)), r_vec, axes=0)
    # rho = x - rp
    # rho_n = np.tensordot(np.ones(len(x_vec)), r_norm, axes=0)

    # NEW VERSION
    r, rplusrn = vectorcalc.calc_r(mid_panel, normal_panel)
    rho, rho_n = vectorcalc.calc_rho(grid, mid_panel, normal_panel)
    # ====================================================================================

    # calculate h, cp, cl, cd and gamma
    h = solve_var(r, rplusrn, length_panel, normal_panel, g, u_inf)

    # calculate potential
    if layer == 'single':
        pot_panel = single(length_panel, h, r)
        v_theta, c_p, c_l = calc_coeff(pot_panel, mid_panel)

        pot_grid = single(length_panel, h, rho)
    elif layer == 'double':
        pot_panel = double(length_panel, h, r, rplusrn)
        v_theta, c_p, c_l = calc_coeff(pot_panel, mid_panel)

        pot_grid = double(length_panel, h, rho, rho_n)
    else:
        raise ValueError('Wrong input argument! Choose either single or double layer instead!')

    return pot_panel, pot_grid, c_p, c_l


# Calculate h and gamma
def solve_var(r, r_n, length_panel, normal_panel, g, u_inf):
    # variable initialization
    r[r == 0] = np.nan
    lambda_ij = cp.do_phi(r, r_n)
    lambda_ij[np.isnan(lambda_ij)] = 0

    # solving the integral matrix
    ds = np.tensordot(np.ones(len(r)), length_panel, axes=0)
    lambda_int = np.multiply(lambda_ij, ds)
    beta = -0.5*np.eye(len(lambda_ij)) - lambda_int

    # calculate the LHS section
    u_ext = np.dot(normal_panel, u_inf)
    lhs = g*np.ones(len(r)) - u_ext

    h = np.dot(np.linalg.inv(beta), lhs)

    # --------------------------------------------------------

    return h


# Single layer
def single(length_panel, h, rho):
    rho[rho == 0] = np.nan

    # calculate potential using single layer
    kai = cp.phi(rho)
    kai[np.isnan(kai)] = 0
    ds_p = np.tensordot(np.ones(len(rho)), length_panel, axes=0)

    res = -np.dot(np.multiply(kai, ds_p), h)

    return res


# Double layer
def double(length_panel, h, rho, rho_n):
    rho[rho == 0] = np.nan

    # calculate potential
    kai = cp.do_phi(rho, rho_n)
    kai[np.isnan(kai)] = 0
    ds_p = np.tensordot(np.ones(len(rho)), length_panel, axes=0)

    res = -np.dot(np.multiply(kai, ds_p), h)

    return res


# Calculate v_tangential, cp and cl from potential at each panel
def calc_coeff(potential, coord):
    x = coord[:, 0]
    y = coord[:, 1]

    # Calculate tangential velocity from potential
    stag_point = np.argmin(np.abs(potential))
    v_theta = np.zeros(len(coord))

    # upper
    for i in range(np.int(stag_point), -1, -1):
        if i == 0:
            v_theta[i] = (potential[i] - potential[i+1]) / (x[i]-x[i+1])
            # (np.arctan2(y[i], x[i]) - np.arctan2(y[i+1], x[i+1]))
        else:
            v_theta[i] = (potential[i-1]-potential[i]) / (x[i-1]-x[i])
            #(np.arctan2(y[i-1], x[i-1]) - np.arctan2(y[i], x[i]))

    # lower
    for i in range(np.int(stag_point+1), len(coord)):
        if i == len(coord)-1:
            v_theta[i] = (potential[i] - potential[i-1]) / (x[i]-x[i-1])
            # (np.arctan2(y[i], x[i]) - np.arctan2(y[i+1], x[i+1]))
        else:
            v_theta[i] = (potential[i+1]-potential[i]) / (x[i+1]-x[i])
            #(np.arctan2(y[i-1], x[i-1]) - np.arctan2(y[i], x[i]))

    #v_theta[:] = np.divide(v_theta, np.linalg.norm(coord, axis=1))

    # Calculate cp from tangential velocity
    c_p = 1 - v_theta**2

    # Calculate cl from cp
    c_pu = c_p[0:stag_point]
    c_pl = c_p[stag_point+1:]
    x_u = x[0:stag_point]
    x_l = x[stag_point+1:]

    c_l = 0
    for i in range(0, len(c_pu)-1):
        c_l += 0.5*(c_pu[i+1] + c_pu[i])*(x_u[i+1] - x_u[i])
    for i in range(0, len(c_pl)-1):
        c_l -= 0.5*(c_pl[i+1] + c_pl[i])*(x_l[i+1] - x_l[i])

    return v_theta, c_p, c_l