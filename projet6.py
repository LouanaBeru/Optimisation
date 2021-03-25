##README
#This code uses operations_Matricielles, available on our website
#Please make sure to save those two in the same directory before running projet6

##MODULES
import operations_Matricielles as om
import operations_Vectorielles as ov
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt 
from math import sqrt, exp
from scipy.optimize import minimize_scalar
from numpy import linalg as LA
from scipy.linalg import hilbert

##CONDITIONS INITIALES
argentDepart = 1000 #argent investi au depart ( en K euros )
nActions = 3 #nbr d'actions
revObj = 1001 #revenu objectif ( en K euros )

##VARIATIONS SUR 1 AN TOUT LES ~18 DU MOIS, VALEUR DE L ACTION A LA CLOTURE
###L OREAL
O = [228.6, 256.8, 253.6, 278.7, 290.7, 281.4, 280.8, 295, 317.4, 306.6, 298.7, 319.1, 326.4]

###RENAULT
R = [16.478, 17.028, 18.998, 22.335, 24.14, 24.685, 23.585, 24.165, 31.48, 37.125, 34.815, 40.135, 39.975]

###MICHELIN
M = [75.06, 90.48, 88.6, 91.5, 94.62, 96.8, 96.14, 95.1, 108.6, 109.6, 109.35, 117.95, 127.05]

vectActions = [O, R, M]

##FONCTIONS UTILES

#Fonction : moy
#Parametres : x : list
#Retourne : moyenne de x
def moy(x):
    m=0
    for i in range(len(x)):
        m+=x[i]
    return m/len(x)

###Evaluation des moyennes
moyO = moy(O)
moyR = moy(R)
moyM = moy(M)

# print(moyO, moyR, moyM)

#Fonction : var
#Parametres : x :list
#Retourne : la variance de x
def var(x):
    variance = 0
    moyX = moy(x)
    for i in x:
        variance += (i - moyX)**2
    return variance / len(x)

###Evaluation des variances
varO = var(O)
varR = var(R)
varM = var(M)

# print(varO, varR, varM)

#Fonction : sigma
#Parametres : x :list
#Retourne : l'ecart-type de x
def sigma(x):
    return sqrt(var(x))


#Fonction : cov
#Parametres : x : list, y : list
#Retourne : covariance de x et y
def cov(x,y):
    if len(x) == len(y):
        covar = 0
        X = moy(x)
        Y = moy(y)
        for i in range(len(x)):
            covar += (x[i] - X) * (y[i] - Y)
        return covar / len(x)
    else :
        print('Les variables ne sont pas de memes dimensions')

###Evaluation des covariances
covOR = cov(O,R)
covOM = cov(O,M)
covMR = cov(M,R)

# print(covOR, covOM, covMR)

###Rendement des actions

#Fonction : rendAction
#Parametres : X : list / cours de l'action
#Retourne : le rendement d'une action
def rendAction(X):
    if X[0] != 0:
        rendement = X[-1] / X[0]
    else:
        rendement = X[-1]
    return rendement 

####L'OREAL
rendO = rendAction(O)

####RENAULT
rendR = rendAction(R)

####MICHELIN
rendM = rendAction(M)

#Vecteur rendement des actions
rendActions = [rendO, rendR, rendM]
print('rendActions = ', rendActions)

##METHODE zero_vect_fun

#J(x) = 1/2 * < Ax ; x > fonction de risque
#E1 = x[0] + x[1] + x[2] - 1
#E2 = x[0] * r[0] + x[1] * r[1] + x[2] * r[2] - R
#E3 = - (x[0] * x[1] + x[1] * x[2] + x[0] * x[2]) / (x[0] * x[1] + x[1] * x[2] + x[0] * x[2])
#x = (proportion investie dans l'Oreal, prop. invest dans Renault, prop. invest dans Michelin)
#On cherche le x qui minimise J

def fun(x):
    return [x[0]*varO + (x[1] * covOR + x[2] * covOM)/2 + x[3] + x[4],
            (x[0] * covOR)/2 + x[1] * varR + (x[2] * covMR)/2 + x[3] + x[4],
            (x[0] * covOM)/2 + (x[1] * covMR)/2 + x[2] * varM + x[3] + x[4],
            x[0]+x[1]+x[2]-1,
            x[0]*rendActions[0] +x[1]*rendActions[1] +x[2]*rendActions[2]-revObj,
            ]

def jacobian(x):
    return np.array([[varO , covOR/2 , covOM/2 , 1, 1],
                     [covOR/2 , varR , covMR/2 , 1, 1],
                     [covOM/2 , covMR/2 , varM , 1, 1],
                     [ 1 , 1 ,  1 , 0, 0],
                     [rendActions[0] , rendActions[1] ,  rendActions[2] , 0, 0],                   
                     ])

sol = sp.optimize.root(fun,[1, 0, 0, 0, 0], jac=jacobian)
for i in range(nActions):
    x1 = np.zeros(nActions)
    x1[i] = max(sol.x[i], 0)
propInvestieZero1 = x1
print('with jacobian=',propInvestieZero1)
print('E1(x) constraint=',propInvestieZero1[0]  + propInvestieZero1[1] + propInvestieZero1[2] - 1)
print('E2(x) constraint=',propInvestieZero1[0]*rendActions[0] + propInvestieZero1[1]*rendActions[1] + propInvestieZero1[2]*rendActions[2]-revObj)

print('---------------------------------------------------------')

sol = sp.optimize.root(fun,[1, 0, 0, 0, 0], method='broyden1')
for i in range(nActions):
    x1 = np.zeros(nActions)
    x1[i] = max(sol.x[i], 0)
propInvestieZero2 = x1
print('broyden1 =',propInvestieZero2)
print('E1(x) constraint=',propInvestieZero2[0]  + propInvestieZero2[1] + propInvestieZero2[2] - 1)
print('E2(x) constraint=',propInvestieZero2[0]*rendActions[0] + propInvestieZero2[1]*rendActions[1] +propInvestieZero2[2]*rendActions[2]-revObj)

##METHODE primal-dual
##Construction de J
####Construction de A
A = [[varO, covOR, covOM],
    [covOR, varR, covMR],
    [covOM, covMR, varM]]

def J(x):
    evaluation = om.produitMatNbr( 0.5, ov.scal( om.produitMat(A,x), x))
    return evaluation

B = ([[1 , 1, 1],
    [rendO, rendR, rendM]])

c = ([[ 1 ],
    [ revObj ]])

invA = om.inverse(A)
transB = om.transposee(B)

def H(p):
    xPrime = invA * transB * p
    evaluation = 1 / 2 * ov.scal( om.produitMat(A ,invA * xPrime), xPrime) + ov.scal(p, om.produitMat(B, xPrime) - c)
    return evaluation

#On resoud une equation du type AX = B avec A = -BA^(-1)B^T et B = c
Atemp = om.produitMat(om.produitMatNbr(-1, B), om.produitMat(invA, transB))
p = np.linalg.solve( Atemp , c)
print('p = ', p)
x = om.produitMat(invA,om.produitMat(transB, p))

for i in range(nActions):
    x2 = np.zeros(nActions)
    x2[i] = max(x[i], 0)

propInvestiePrimal = x2

print('propInvestiePrimal', propInvestiePrimal)

##METHODE UZAWA
###########################################
def func(ndim, x):
    alp1=0.
    alp2=0.
    alp3=0.
    print('x =',x)
    print('A =', A)
    print('Ax =',om.produitMat(A,x))
    cost = 0.5 * ov.scal( om.produitMat(A,x), x)
    # print('cost = ', cost)
    cons1 = x[0]+x[1]+x[2]-1
    cons2 = x[0]*rendActions[0] +x[1]*rendActions[1] +x[2]*rendActions[2]-revObj
    cons3 = x[0]*x[1] / x[0]*x[1] - x[1]*x[2] / x[1]*x[2]
    f=cost + alp1 * abs(cons1) + alp2 * abs(cons2) + alp3 * abs(cons3)
    return f
###########################################
def funcp(ndim, x):
    fp=[x[0]*varO + (x[1] * covOR + x[2] * covOM)/2 + x[3] + x[4] * rendActions[0] ,
        (x[0] * covOR)/2 + x[1] * varR + (x[2] * covMR)/2 + x[3] + x[4] * rendActions[1] ,
        (x[0] * covOM)/2 + (x[1] * covMR)/2 + x[2] * varM + x[3] + x[4] * rendActions[2],
        -0.2*(x[0]+x[1]+x[2]-1),
        -0.2*(x[0]*rendActions[0] +x[1]*rendActions[1] +x[2]*rendActions[2]-revObj),
        -0.2*(x[0]*x[1] / x[0]*x[1] - x[1]*x[2] / x[1]*x[2])
        ]
    return fp
nbgrad=10000
eps=1.e-6
epsdf=0.001
ndim=5
idf=0
ro0=0.01

#for igc in [0,1]:
for igc in [0]:

    ro=ro0
    it=[]
    history=[]
    historyg=[]
    
    for ii in range(nbgrad):
        it=it+[ii+1]
        history=history+[0]
        historyg=historyg+[0]
        
    dfdx=np.zeros((ndim))
    xmax=np.ones((ndim))*20
    xmin=-np.ones((ndim))*20
    x=np.ones((ndim))
   
    dfdx=np.zeros((ndim))
    d=dfdx

    crit=1
    itera=-1
    while(itera<nbgrad and crit>eps):
        itera+=1
        
        dfdx0=dfdx
        
        if(idf==1):
            for i in range(0, ndim):
                x[i]=x[i]+epsdf
                fp=func(ndim, x3)
                x[i]=x[i]-2*epsdf
                fm=func(ndim, x)
                x[i]=x[i]+epsdf
                dfdx[i]=(fp-fm)/(2*epsdf)
        elif(idf==0):
            dfdx=funcp(ndim,x)
        
        gg=0
        for j in range(0, ndim):
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
            
        f=func(ndim, [x[0], x[1], x[2]] )
        history[itera]=f
        historyg[itera]=gg
        g1=dfdx[0]
        g2=dfdx[1]
        g3=dfdx[2]
        xnoj=np.sqrt(g1**2+g2**2+g3**2)        
        gc11 = 1;
        gc12 = 1;
        gc13 = 1;
        gc21 = rendActions[0];
        gc22 = rendActions[1];
        gc23 = rendActions[2];
        gc31=0;
        gc32=0;
        gc33=0;
        xnoc1=np.sqrt(gc11**2+gc12**2+gc13**2)
        xnoc2=np.sqrt(gc21**2+gc22**2+gc23**2)
        xnoc3=np.sqrt(gc31**2+gc32**2+gc33**2)
        ps1=(gc11*g1+gc12*g2+gc13*g3)/xnoc1
        ps2=(gc11*g1+gc12*g2+gc13*g3)/xnoc2
        ps3=(gc11*g1+gc12*g2+gc13*g3)/xnoc3
        g1-=ps1*gc11/xnoc1+ps2*gc21/xnoc2+ps3*gc31/xnoc3
        g2-=ps1*gc12/xnoc1+ps2*gc22/xnoc2+ps3*gc32/xnoc3
        g3-=ps1*gc13/xnoc1+ps2*gc23/xnoc2+ps3*gc33/xnoc3
        crit=abs(g1)+abs(g2)+abs(g3)
      
#        if(abs(g1)+abs(g2)<eps):    #critere d'arret 

    h1=abs(history[0])
    hg1=abs(historyg[0])

    for iter in range(0, itera):
        history[iter]=history[iter]/h1
        historyg[iter]=historyg[iter]/hg1

    # print("igc=",igc)
    # if igc==0:
    #     plt.plot(it[:itera],history[:itera], color='red', label='GD')
    # if igc==1:
    #     plt.plot(it[:itera],history[:itera], color='green',label='CG')

for i in range(3):
    x3 = np.zeros(nActions)
    x3[i] = max(x[i] , 0)

propInvestieUzawa = x3
print('iterations=',itera)
print('convergence criteria=',crit)
print('uzawa (x,p)=', x3)
print('target(x,p)= [0.16754469  0.55409219 -0.325321   -0.10518421]')
print(x3[0]+x3[1]+x3[2]-1)
print(x3[0]*rendActions[0] +x3[1]*rendActions[1] +x3[2]*rendActions[2]-revObj)
# print(x[0]*x[1] / x[0]*x[1] - x[1]*x[2] / x[1]*x[2])
# plt.legend()
# plt.show()

##RENDEMENT DU PORTEFEUILLE

#Fonction : rendPortefeuille
#Parametres : n : int / nbr d'actions, p : list / proportions investies dans chaque action, r : list / rendement de chaque action
#Retourne : le rendement du portefeuille
def rendPortefeuille(N, p, r):
    rPortefeuille = 0
    for i in range(N):
        rPortefeuille += p[i] * r[i]
    return rPortefeuille

propInvestie = [propInvestieZero2, propInvestiePrimal, propInvestieUzawa]
rendement = rendPortefeuille(nActions, propInvestie, rendActions )

print('rendement = ',rendement)
