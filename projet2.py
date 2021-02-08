from math import exp 
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar
from numpy import linalg as LA
from scipy.linalg import hilbert
import functools

nbgrad=100 # Nombre de Graduation sur la courbe

graduations=list(range(nbgrad)) # Graduation pour l'affichage de la courbe, tout les nombre de 0 a nbgrad

epsilon=0.001
ndim=50
deriveDefinie=False #True si la dérivée de f est définie, False sinon
dist = [] #distance entre le x actuel et le x_cible

#Définition x_cible
xCible = np.ones(ndim)

# #Définition A et b_cible
# A = sp.linalg.hilbert(ndim)
# bCible = np.matmul(A, xCible)

# #Calcul du conditionnement de A
# cond = np.linalg.cond(A, p =1)
# print('Le conditionnement de la matrice est :', cond)

# def J(x): #matrice associée à f
#     return np.matmul(A, x)

# def Jp(x): #matrice associée à f'
#     return A

# Empeche d'aller en dessous de min et au dessus de max
def clamp(val, minval, maxval): 
    return max(min(val, maxval), minval)

### PRECOMPUTE DIVIDERS ####
div = []
for i in range(1, ndim+1): #lignes
    valdiv = 0
    for j in range(1, ndim + 1):
        valdiv += 1/(i+j-1)
    div.append(valdiv)
div = np.asarray(div)

def norm2(value_list):
    return np.sqrt(sum(value_list ** 2)) # Racine de la somme des valeurs mises au carré

def func(x):
    return sum(x * div)

ro0=0.01
for igc in [0, 1]:
    ro=ro0
    history=np.zeros(nbgrad)  # On Recommence a zero
    
    gradiant=np.zeros(ndim)
    xmax=4
    xmin=0.01
    x=np.ones(ndim)*0.2
    
    d=gradiant.copy()
    
    for itera in range(nbgrad): 
        gradiant0=gradiant.copy()
        
        if not deriveDefinie:
            for i in range(ndim):
                x[i] += epsilon
                fp = func(x)
                x[i] -= 2 * epsilon
                fm = func(x)
                x[i] += epsilon
                gradiant[i] = (fp-fm) / (2*epsilon)

        else:
            gradiant=funcp(x)
        
        if igc == 0:
            for j in range(ndim):
                d[j] = -gradiant[j]
      
        elif igc == 1:
            xnum=0
            xden=0
            for j in range(ndim):
                xnum = xnum + gradiant[j] * (gradiant[j] - gradiant0[j])
                xden = xden + gradiant[j] ** 2

            beta=0
            if(xden>1.e-30):
                beta = max(0,xnum/xden) # 0 ou plus
                
            for j in range(ndim):
                # print(beta, d[j], beta * d[j])
                d[j]= -gradiant[j] + beta * d[j]
            
        for i in range(ndim):
            x[i] = x[i] + ro * d[i]
            x[i] = clamp(x[i], xmin, xmax) # Empeche de dépasser les bornes

        if igc == 1:
            dist.append(func(x) - func(xCible))

        f=func(x)
        history[itera]=f
  
        if (itera > 2 and history[itera-1] > f):
            ro = min(ro*1.25, 100*ro0)
        else:
            ro = max(ro*0.6, 0.01*ro0)

        h1 = abs(history[1])
        
    for itera in range(0, nbgrad):
        history[itera]=np.log10(history[itera]/h1)

    print("igc=",igc)
    plt.figure(1)
    if igc == 0:
        plt.plot(graduations, history, color='red', label='GD')
        plt.legend()
    if igc == 1:
        plt.plot(graduations, history, color='green',label='CG')
        plt.legend()

plt.figure(2)
plt.plot(graduations, dist, label ='Evolution de la distance entre x et x_cible', color = (0,0,1, 1))
plt.legend()
plt.show()

print(x)
