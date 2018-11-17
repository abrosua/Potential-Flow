# Calculate phi for 2D
# phi(r) = -ln|r| / (2*pi)
# note that r and r_n must be an array contains of scalar r, instead of vector r!
import numpy as np

def phi(r):
    res = -np.log(np.abs(r)) / (2*np.pi)
    return res


def do_phi(r, rplusrn):
    res = np.divide((phi(rplusrn) - phi(r)), rplusrn)
    return res
