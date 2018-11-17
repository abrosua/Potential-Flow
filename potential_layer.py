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
    r,rplusrn = vectorcalc.calc_r(mid_panel,normal_panel)
    rho,rho_n = vectorcalc.calc_rho(grid,normal_panel)
    # ====================================================================================

    # calculate h, cp, cl, cd and gamma
    h, c_p, c_l = solve_var(r, rplusrn, length_panel, mid_panel, g, u_inf)

    # calculate potential
    if layer == 'single':
        potential = single(length_panel, h, rho)
    elif layer == 'double':
        potential = double(length_panel, h, rho, rho_n)
    else:
        raise ValueError('Wrong input argument! Choose either single or double layer instead!')

    return potential, c_p, c_l


# Calculate h and gamma
def solve_var(r, r_n, length_panel, mid_panel, g, u_inf):
    # variable initialization
    r[r == 0] = np.nan
    lambda_ij = cp.do_phi(r, r_n)
    lambda_ij[np.isnan(lambda_ij)] = 0

    # solving the integral matrix
    ds = np.tensordot(np.ones(len(r)), length_panel, axes=0)
    lambda_int = np.multiply(lambda_ij, ds)
    beta = -0.5*np.eye(len(lambda_ij)) - lambda_int

    # calculate the LHS section
    u_ext = np.dot(mid_panel, (u_inf))
    lhs = g*np.ones(len(r)) - u_ext

    h = np.dot(np.linalg.inv(beta), lhs)

    # --------------------------------------------------------

    # calculate tangential velocity (V_i = V_s + U_inf.n)
    temp_v = np.dot(beta, h)
    v_i = temp_v + u_ext

    # calculate cp = 1 - (v_i/|U_inf|)^2
    c_p = 1 - np.divide(v_i, np.linalg.norm(u_inf))**2

    # --------------------------------------------------------

    # calculate cl and cd from cp
    c_l = np.dot(c_p, length_panel)

    return h, c_p, c_l


# Single layer
def single(length_panel, h, rho):
    # calculate potential using single layer
    kai = cp.phi(rho)
    ds_p = np.tensordot(np.ones(len(rho)), length_panel, axes=0)

    res = -np.dot(np.multiply(kai, ds_p), h)

    return res


# Double layer
def double(length_panel, h, rho, rho_n):
    # calculate potential
    kai = cp.do_phi(rho, rho_n)
    ds_p = np.tensordot(np.ones(len(rho)), length_panel, axes=0)

    res = -np.dot(np.multiply(kai, ds_p), h)

    return res
