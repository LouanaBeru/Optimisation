##MODULES
from math import exp
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import random
from array import array

##PARAMETRAGE
y = [1, 1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]
x= [2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]
n = len(y)
ro0 = 3*10**(-3)
cadr = 1000 #découpe en "card" point [0,1]


##FONCTIONS UTILES

#Fonction : prolisint
#Parametres : l : list, a : int
#Retourne : l * a
def prolisint(l,a):
    pro=[]
    for i in range(0, n):
        pro = pro + [l[i]*a]
    return pro

#Fonction : sum
#Parametres : x : list, y : list
#Retourne : x + y
def sum(x,y):
    som = []
    for i in range(0, n):
        som = som + [x[i] + y[i]]
    return som

##AUTRES FONCTIONS

#Fonction : scal
#Parametres : x : list, y : list
#Retourne : < x ; y > 
def scal(x, y):
    sxy = 0
    for i in range(0,n):
        sxy = sxy + x[i] * y[i]
    return sxy

#Fonction : norme2
#Parametres : x : list
#Retourne : (||x||_2) ** 2
def norme2(x):
    n2=0
    for i in range(0, n):
       n2 = n2 + x[i]**2 
    return n2

##DEFINITION DE f(x)

#f(x) = <x,y> exp(-||x||^2)
def f(x,y):
    f=scal(x,y)*exp(-norme2(x))
    return f

def J(x):
    J = 0
    for i in range(n-1):
        J = J + (x[i]**2 )/ (2(i+1))
    return J



##GRADIENT

#Fonction : dfdxi
#Parametres : x : list, y : list, i : int
#Renvoie : la derivee partielle de f suivant x[i]
def dfdxi(x, y, i):
    fxi = exp(-norme2(x))*(y[i]-2*x[i]*scal(x,y))
    return fxi

#Fonction : gradi
#Parametres : x : list
#Renvoie : grad(f)
def gradi(x):
    grad=[] 
    for j in range(0, n):
        grad=grad+[dfdxi(x,y,j)]
    return grad

##NORMALISATION DU GRADIENT

#Fonction : gradnor
#Parametres : x : list
#Retourne : la gradient de f normalise
def gradnor(x):
    gradnorm = prolisint(  gradi(x)  ,  1 / sqrt( norme2(gradi(x))))
    return(gradnorm)

#############################################################################################
##PARAMETRAGE
x0 =[0, 0, 0 ,0 , 0, 0, 0, 0 ,0 ,0]
eps=0.0000001 #epsilon
ro = 0.03 #pas de base
xi = x0
gx = [f(x0,y)]
absy = [0]
nb = 0

##RECHERCHE D'UN POINT FIXE
for it in range(1, 10000):
    
    if ( norme2(gradi(x))*ro != 0) and (f( sum( xi, prolisint(gradnor(xi), -ro) ) , y)  <  f( xi, y)): #On regarde si elle est toujours décroissante
        if (f( sum( xi, prolisint(gradi(xi), -ro) ) , y)  <  f( xi, y)):
            gx = gx + [f(xi,y)]                     #liste des f(xi)
            nb = nb + 1                             #compteur
            absy = absy + [nb*ro]          #liste des absysse
            xi = sum( xi, prolisint(gradnor(xi),-ro))          #passage à xi+1


#############################################################################################
##AFFICHAGE
print('il y a eu', nb, 'étapes pour arriver à f = zéro qui est pour Xmin =',xi)
plt.figure(1)
plt.xlabel(u'$Distance parcourue$', fontsize=26)
plt.ylabel(u'$F(x)$', fontsize=26, rotation=90)
plt.title(u'F(x) en fonction du chemin descendant')
plt.plot(absy, gx)

plt.figure(2)
plt.plot(x, np.log10(abs(f-J)))
plt.show()

