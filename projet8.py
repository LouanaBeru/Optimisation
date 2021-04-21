##MODULES
import numpy as np

##SIMPLEX METHOD
###DEFINITION OF THE PROBLEM
#J(x1, x2, x3) = -5x1 - 4x2 - 6x3
#x1 - x2 + x3 \leq 20
#3x1 + 2x2 + 4x3 \leq 42
#3x1 +2x2 \leq 30
#0 \leq x1, 0 \leq x2, 0 \leq x3

A = np.mat([
    [ 1, -1, 1, 20],
    [3, 2, 4, 42],
    [3, 2, 0, 30],
    [-5, -4, -6, 1]
])

###TURN INTO A MAXIMIZATION PROBLEM
AT = A.transpose()

##
B = np.mat([
    [-AT[-1][1], -AT[-1][2], AT[-1][3], AT[-1][4], 0],
    [AT[1][1], AT[1][2], AT[1][3], 1, -5],
    [AT[2][1], AT[2][2], AT[2][3], 1, -4],
    [AT[3][1], AT[3][2], AT[3][3], 1, -6]
])