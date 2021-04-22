##MODULES
import numpy as np

##SIMPLEX METHOD
###DEFINITION OF THE PROBLEM
#Minimize
#J(x1, x2, x3) = -5x1 - 4x2 - 6x3
#With
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
print(AT)
linesAT = np.shape(AT)[0]
columnsAT = np.shape(AT)[1]

#New problem :
#With
#y1 + 3y2 + 3y3 \leq -5
#-y1 +2y2 + 2y3 \leq -4
#y1 + 4y2 \leq -6
#Maximise
#P = 20y1 + 42y2 + 30y3

###EQUALITY PROBLEM
#Equivalent to
#-20y1 - 42y2 - 30y3 + P = 0
#y1 + 3y2 + 3y3 + x1 = -5
#-y1 +2y2 + 2y3 + x2 = -4
#y1 + 4y2 + x3 = -6

linesB = linesAT
columnsB = columnsAT + 4

B = np.zeros((linesB, columnsB))

#Here, 
#B = [[ 1, 3, 3, 1, 0, 0, 0, -5],
# [-1, 2, 2, 0, 1, 0, 0, -4],
# [1, 4, 0, 0, 0, 1, 0, 0, -6],
# [-20, -42, -30, 0, 0, 0, 1, 0]]

for i in range(linesB):

    if 0 <= i <= linesAT-2:
        for j in range(columnsAT-1):
            B.itemset((i, j), AT.item(i, j))
        for j in range(columnsAT-1,columnsB-1):
            if i+3 == j:
                B.itemset((i, j), 1)   
        B.itemset((i, columnsB-1), AT.item(i, columnsAT-1))  

    if i == linesB-1:
        for j in range(columnsAT-1):
            B.itemset((i, j), -AT.item(i,j))
        B.itemset((i, columnsB-2), 1)        
print(B)

###SOLVING

entrant = []
for j in range(columnsAT-1):
    entrant.append(B.item(linesB-1, j))

entrant = np.asarray(entrant)
minimum = min(entrant)

for i in range(len(entrant)):
    if entrant[i] == minimum:
        rangMinEntrant = i

sortant = []
for i in range(linesAT-1):
    sortant.append(B.item(i,columnsB-1)/B.item(i, 0))

minimum = min(sortant)

for i in range(len(sortant)):
    if sortant[i] == minimum:
        rangMinSortant = i

###Si on diagonalise A|I, on obtient D|P













# B = np.mat([
#     [-AT[-1][1], -AT[-1][2], AT[-1][3], AT[-1][4], 0],
#     [AT[1][1], AT[1][2], AT[1][3], 1, -5],
#     [AT[2][1], AT[2][2], AT[2][3], 1, -4],
#     [AT[3][1], AT[3][2], AT[3][3], 1, -6]
# ])