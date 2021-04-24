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

linesA = np.shape(A)[0]

###TURN INTO A MAXIMIZATION PROBLEM
AT = A.transpose()
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
print('#####################')   
print('MAXIMIZATION PROBLEM')  
print('B=')
print(B)
print('#####################')
print('DIAGONALIZATION')

###DIAGONALAZING

for tour in range(linesA-1):
    #Checks the minimum value on the last line of B between the columns 0 and columnsAT-2
    entrant = []
    for j in range(columnsAT-1):
        entrant.append(B.item(linesB-1, j))

    entrant = np.asarray(entrant)
    minimum = min(entrant)
    print('minimum on the last line:',minimum)

    #Gives his column
    for i in range(len(entrant)):
        if entrant[i] == minimum:
            rangMinEntrant = i
            i = len(entrant)
    print('is at the column number:',rangMinEntrant)

    #Checks the minimum value on the last column of B between the lines 0 and linesB-1
    sortant = []
    for i in range(linesAT-1):
        if B.item(i, rangMinEntrant) != 0:
            sortant.append(B.item(i,columnsB-1)/B.item(i, rangMinEntrant))
        else:
            sortant.append(B.item(i,columnsB-1))

    sortant = np.asarray(sortant)
    minimum = min(sortant)
    print('minimum on the last column:',minimum)

    #Gives his line
    for i in range(len(sortant)):
        if sortant[i] == minimum:
            rangMinSortant = i
            i = len(sortant)
    print('is at the line number:',rangMinSortant)

##Operating on lines
    print('#####################')
    print('OPERATING ON LINES')
    #if the pivot is zero, the program has to stop
    if B.item(rangMinSortant, rangMinEntrant) == 0:
        tour = linesA
    else:
        print('pivot is B[',rangMinSortant,'][', rangMinEntrant,']=',
        B.item(rangMinSortant,rangMinEntrant))
        print()
        for i in range(linesB):
            if i != rangMinSortant: #Changes every lines except the pivot's line
                coeff = B.item(i,rangMinEntrant) #We have to copy it,
                #because after 1 loop it will be changed as 0
                #Li become Li - B[i][pivot's column] / pivot * L_of_the_pivot
                for j in range(columnsB): #Goes through every column
                    B.itemset((i, j), B.item(i,j) 
                    - coeff/B.item(rangMinSortant, rangMinEntrant) 
                    * B.item(rangMinSortant, j) )
                print('#####################')
                print('L', i, 'devient L', i, 
                '- B[', i, '][', rangMinEntrant, ']/B[', rangMinSortant, '][', rangMinEntrant, 
                '] * L', rangMinSortant)
                print('B=')
                print(B)
print('end')
print('B=')
print(B)
#Can find the final answer on the last line of B
print('Final solution is :')
print('x0 =', B.item(linesB-1, columnsB - 5),
 '; x1 =', B.item(linesB-1, columnsB - 4 ),
  '; x2 =', B.item(linesB-1, columnsB - 3), '; J =',
   B.item(linesB-1, columnsB -1))

