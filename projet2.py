from math import exp 
import matplotlib.pyplot as plt 
import numpy as np
import scipy as sp
from scipy.optimize import minimize_scalar
from numpy import linalg as LA
from scipy.linalg import hilbert

nbgrad=500
epsdf=0.001
ndim=50
idf=1 #0 si la dérivée de f est définie, 1 sinon

#Définition x_cible
xCible = np.ones((ndim))

#Définition A et b_cible
A = sp.linalg.hilbert(ndim)
bCible = np.matmul(A, xCible)

#Calcul du conditionnement de A
cond = np.linalg.cond(A, p =1)
print('Le conditionnement de la matrice est :', cond)

def J(x): #matrice associée à f
    J = np.matmul(A, x)
    return J

def Jp(x): #matrice associée à f'
    Jp = A
    return Jp

def func(x):
    f = 0
    for i in range(1, ndim+1): #lignes
        for j in range (1, ndim+1): #colonnes
            f = f + x[i-1] / (i+j-1) #On applique la matrice de Hilbert
    return f

def funcp(x):
    fp = 0
    for i in range(1, ndim+1): #lignes
        for j in range (1, ndim+1): #colonnes
            fp = fp + 1 / (i+j-1) 
    return fp

for igc in [0,1]:
    ro0=0.01
    ro=ro0
    it=[]
    history=[]
    historyg=[]
    
    for ii in range(0, nbgrad):
        it=it+[ii+1]
        history=history+[0]
        historyg=historyg+[0]
        
    dfdx=np.zeros((ndim))
    xmax=np.ones((ndim))*2
    xmin=-np.ones((ndim))*2
    x=np.ones((ndim))*0.1
   
    dfdx=np.zeros((ndim))
    d=dfdx
    
    for itera in range(nbgrad): 
        dfdx0=dfdx
        
        if(idf==1):
            for i in range(ndim):
                x[i]=x[i]+epsdf
                fp=func(x)
                x[i]=x[i]-2*epsdf
                fm=func(x)
                x[i]=x[i]+epsdf
                dfdx[i]=(fp-fm)/(2*epsdf)
        elif(idf==0):
            dfdx=funcp(x)
        
        gg=0
        for j in range(ndim):
            gg=gg+dfdx[j]**2
        
        if igc==0:
            for j in range(0, ndim):
                d[j]=-dfdx[j]
                
        if igc==1:
            xnum=0
            for j in range(0, ndim):
                xnum=xnum+dfdx[j]*(dfdx[j]-dfdx0[j])
            xden=0
            for j in range(0, ndim):
                xden=xden+dfdx[j]**2
            beta=0
            if(xden>1.e-30):
                beta=max(0,xnum/xden)
                
            for j in range(0, ndim):
                d[j]=-dfdx[j]+beta*d[j]
            
        for i in range(0, ndim):
            x[i]=x[i]+ro*d[i]
            x[i]=max(min(x[i], xmax[i]), xmin[i])
            
        f=func(x)
        history[itera]=f
        historyg[itera]=gg

####################################################
#two ways to define the step size
#incomplete linesearch 
#            ro1=3
#            beta=0.5        
#            for i in range(0, 10):
#                xtest=x+ro1*d
#                ftest=func(ndim,xtest)
#                if(ftest > f-ro1/2*gg):
#                    ro1=ro1*beta
#                ro=ro1        
#heuristic for ro tuning        
        if (itera >2 and history[itera-1] > f):
            ro=min(ro*1.25, 100*ro0)
        else:
            ro=max(ro*0.6, 0.01*ro0)
#####################################################

        h1=abs(history[1])
        hg1=historyg[1]

    for itera in range(0, nbgrad):
        history[itera]=np.log10(history[itera]/h1)
        historyg[itera]=historyg[itera]/hg1

    plt.figure(1)
    print("igc=",igc)
    if igc==0:
        plt.plot(it,history, color='red', label='GD')
    if igc==1:
        plt.plot(it,history, color='green',label='CG')

# plt.figure(2)    
# it1=[]
# for j in range(nbgrad):
#    it1=it1+[1.0/(it[j])]
# plt.plot(it, np.log10(it1), color='blue')
# it2=[]
# for j in range(nbgrad):
#    it2=it2+[it1[j]**2]
# plt.plot(it, np.log10(it2), color='blue')

plt.show()

print(x)
print(-np.ones((ndim))/np.sqrt(2.)/np.sqrt(2.))
