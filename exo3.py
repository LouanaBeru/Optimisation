from math import exp #on importe la fonction exp
from math import sqrt
import matplotlib.pyplot as plt #module permettant de tracer des courbes
import numpy as np
import random
from array import array

####PARAMETRAGE####
y = [1, 1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]
x= [2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ]
n = len(y)
ro0 = 3*10**(-3)
cadr = 1000 #découpe en "card" point [0,1]


####FONCTION UTILE#####
def prolisint(x,a): #produit entre une liste et un nombre
    pro=[]
    for i in range(0, n):
        pro = pro + [x[i]*a]
    return pro

def sum(x,y):  #Somme entre deux listes
    som = []
    for i in range(0, n):
        som = som + [x[i] + y[i]]
    return som

####FONCTION POUR DEFINIR f(x)#####
def scal(x, y): #produit de scalaire de x,y
    sxy = 0
    for i in range(0,n):
        sxy = sxy + x[i] * y[i]
    return sxy

def norme2(x): #norme au carré
    n2=0
    for i in range(0, n):
       n2 = n2 + x[i]**2 
    return n2

#### DEFINITION DE f(x) #####
def f(x,y): #f(x) = <x,y> exp(-||x||^2)
    f=scal(x,y)*exp(-norme2(x))
    return f

#### GRADIENT ####
def dfdxi(x, y, i): #dérivée partielle dans la i ème coordonnée
    fxi = exp(-norme2(x))*(y[i]-2*x[i]*scal(x,y))
    return fxi

def gradi(x): #gradient
    grad=[] 
    for j in range(0, n):
        grad=grad+[dfdxi(x,y,j)]
    return grad

#### NORMALISATION DU GRADIENT ####

def gradnor(x):
    gradnorm = prolisint(  gradi(x)  ,  1 / sqrt( norme2(gradi(x))))  ###division du gradient par ça norme
    return(gradnorm)

#### PARAMETRAGE ####

x0 =[0, 0, 0 ,0 , 0, 0, 0, 0 ,0 ,0]
eps=0.0000001 #epsilon
ro = 0.03 #pas de base
xi = x0
gx = [f(x0,y)]
absy = [0]
nb = 0

#### RECHERCHE D'UN POINT FIXE ####
for it in range(1, 10000):
    
    if ( norme2(gradi(x))*ro != 0) and (f( sum( xi, prolisint(gradnor(xi), -ro) ) , y)  <  f( xi, y)): #On regarde si elle est toujours décroissante
        if (f( sum( xi, prolisint(gradi(xi), -ro) ) , y)  <  f( xi, y)):
            gx = gx + [f(xi,y)]                     #liste des f(xi)
            nb = nb + 1                             #compteur
            absy = absy + [nb*ro]          #liste des absysse
            xi = sum( xi, prolisint(gradnor(xi),-ro))          #passage à xi+1



####AFFICHAGE####
print('il y a eu', nb, 'étapes pour arriver à f = zéro qui est pour Xmin =',xi)
plt.figure()
plt.xlabel(u'$Distance parcourue$', fontsize=26)
plt.ylabel(u'$F(x)$', fontsize=26, rotation=90)
plt.title(u'F(x) en fonction du chemin descendant')
plt.plot(absy, gx)
plt.show()