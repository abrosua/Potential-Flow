# Potential-Flow
Potential flow solver for single and double layer

main.py is the main program to call each subroutine

Functions:
  1. import_airfoil.py  : To import airfoil coordinate from .txt file and create each panels.
  2. calc_phi.py        : To perform numerical calculation of phi and d(phi)/d(n).
  3. potential_layer.py : To calculate the potential equation for single and double layer.
  4. gridgen_fcn.py     : To generate grid for the airfoil domain.
  5. post_pro.py        : To generate results post-processing
