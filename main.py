import import_airfoil as ia
import potential_layer as pl
import gridgen_fcn as gg
import numpy as np
import matplotlib.pyplot as plt

filename = "airfoil_coord.txt"

mid_panel, normal_panel, length_panel, airfoil = ia.importer(filename)

# input grid
domsize = 2
grid = gg.pointsgen(filename, domsize)

# input boundary condition
u_inf = np.array([5, 0])
g = 0

# calculate potential
layer = 'single'
potential, c_p, c_l = pl.pot(mid_panel, normal_panel, length_panel, g, u_inf, layer, grid)

# post-processing
plt.plot(mid_panel[:, 0], c_p)
plt.show()

print(c_l)
