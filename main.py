import import_airfoil as ia
import potential_layer as pl
import gridgen_fcn as gg
import numpy as np
import matplotlib.pyplot as plt

filename = "4412.txt"

mid_panel, normal_panel, length_panel, airfoil = ia.importer(filename)
tes = np.linalg.norm(normal_panel, axis=1)

# input grid
domsize = 2
npoints = 10
grid = gg.pointsgen(filename, domsize, npoints)

# input boundary condition
u_inf = np.array([1, 0])
g = 0

# calculate potential
layer = 'double'
pot_panel, pot_grid, c_p, c_l = pl.pot(mid_panel, normal_panel, length_panel, g, u_inf, layer, grid)

# post-processing
plt.plot(mid_panel[:, 0], -c_p)
plt.show()

print(c_p)
print(c_l)
