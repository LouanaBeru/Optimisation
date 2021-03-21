import numpy as np
import matplotlib.pyplot as plt
import random
import scipy as sp
from scipy.optimize import minimize_scalar
from numpy import linalg as LA
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import exp, sqrt
from mpl_toolkits.mplot3d import Axes3D 
from mpl_toolkits import mplot3d



epsi=0.0001

def prolisint(x,a): #produit entre une liste et un nombre
    pro=[]
    for i in range(2):
        pro = pro + [x[i]*a]
    return pro

def norme2(x): #norme au carré
    n2=0
    for i in range(2):
       n2 = n2 + x[i]**2 
    return sqrt(n2)

def scal(x, y): #produit de scalaire de x,y
    sxy = 0
    for i in range(2):
        sxy = sxy + x[i] * y[i]
    return sxy

def sum(x,y):  #Somme entre deux listes
    som = []
    for i in range(2):
        som = som + [x[i] + y[i]]
    return som

def fun(x,y):
    return -(np.log10(x)+np.log10(y)+np.log10(1-x)+np.log10(1-y))

def funa(x):
    return np.log10(x[0])+np.log10(x[1])+np.log10(1-x[0])+np.log10(1-x[1])

def fun1d(x):
    return -(np.log10(x)+np.log10(1-x))


fig = plt.figure(1)
####################


ax = fig.gca(projection='3d')

# Make data.
X = np.arange(0, 1, 0.001)
Y = np.arange(0, 1, 0.001)
X, Y = np.meshgrid(X, Y)
Z = fun(X,Y)

# Plot the surface.
ax.plot_surface(X, Y, Z)




############################



a0 = random.choices(range(1, 999), k=2)
a0 = prolisint(a0,1/1000)
print('point de départ=',a0)

def grad(x):
    return [ fun1d(x[0]+0.001)+ fun1d(x[0]-0.001)- 2*fun1d(x[0]) , fun1d(x[1]+0.001)+ fun1d(x[1]-0.001)- 2*fun1d(x[1])]

grada= prolisint(grad(a0),-1)
histx=[a0[0]]
histy=[a0[1]]
histf=[funa(a0)]
histg=[grad(a0)]
sca=[]
it=1
itl=[]

print(histg)

while (norme2(grada)>epsi) and (it<10000):
    itl+=[it]
    i=0
    while(funa(a0)>funa(sum(a0,prolisint(grada,100))) and i<10000):
        a0= sum(a0,prolisint(grada,100))
        histx+=[a0[0]]
        histy+=[a0[1]]       
        histf.append([funa(a0)])
        i+=1
        if i==10000:
            it=10000
            print('problème, veuiller essayer à nouveau')

    sca+= [scal(grada,prolisint(grad(a0),-1))]
    grada=prolisint(grad(a0),-1)
    it+=1

print(sca)

ax.plot3D(histx, histy, histf, 'red')


 

#########################
fig = plt.figure(2)

plt.plot(itl, sca, color='green')

plt.legend()
plt.show()