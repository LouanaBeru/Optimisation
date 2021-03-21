##README
#This code uses operations_Matricielles and operations_Vectorielles, available on our website
#Please make sure to save those three in the same directory before running projet6

##MODULES
import operations_Matricielles as om
import operations_Vectorielles as ov
from scipy import optimize
import numpy as np
from math import sqrt

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
print(rendActions)

##METHODE zero_vect_fun

#J(x) = 1/2 * < Ax ; x > fonction de risque
#E1 = x[0] + x[1] + x[2] - 1
#E2 = x[0] * r[0] + x[1] * r[1] + x[2] * r[2] - R
#x = (proportion investie dans l'Oreal, prop. invest dans Renault, prop. invest dans Michelin)
#On cherche le x qui minimise J

def fun(x):
    return [x[0]*varO + (x[1] * covOR + x[2] * covOM)/2 + x[3] + x[4],
            (x[0] * covOR)/2 + x[1] * varR + (x[2] * covMR)/2 + x[3] + x[4],
            (x[0] * covOM)/2 + (x[1] * covMR)/2 + x[2] * varM + x[3] + x[4],
            x[0]+x[1]+x[2]-1,
            x[0]*rendActions[0] +x[1]*rendActions[1] +x[2]*rendActions[2]-revObj
            ]

def jacobian(x):
    return np.array([[varO , covOR/2 , covOM/2 , 1, 1],
                     [covOR/2 , varR , covMR/2 , 1, 1],
                     [covOM/2 , covMR/2 , varM , 1, 1],
                     [ 1 , 1 ,  1 , 0, 0],
                     [rendActions[0] , rendActions[1] ,  rendActions[2] , 0, 0]                     
                     ])

sol = optimize.root(fun,[1, 0, 0, 0, 0], jac=jacobian)
propInvestieZero1 = sol.x
print('with jacobian=',propInvestieZero1)
print('E1(x) constraint=',propInvestieZero1[0]  + propInvestieZero1[1] + propInvestieZero1[2] - 1)
print('E2(x) constraint=',propInvestieZero1[0]*rendActions[0] + propInvestieZero1[1]*rendActions[1] + propInvestieZero1[2]*rendActions[2]-revObj)

print('---------------------------------------------------------')

sol = optimize.root(fun,[1, 0, 0, 0, 0], method='broyden1')
propInvestieZero2 = sol.x
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
    evaluation = 1 / 2 * ov.scal( om.produitMat(A,x), x)
    return evaluation

B = ([[1 , 1, 1],
    [rendO, rendR, rendM]])

c = ([[ 1 ],
    [ revObj ]])

invA = om.inverse(A)
transB = om.transpose(B)

def H(p):
    xPrime = invA * transB * p
    evaluation = 1 / 2 * ov.scal( om.produitMat(A ,invA * xPrime), xPrime) + ov.scal(p, om.produitMat(B, xPrime) - c)
    return evaluation

#On resoud une equation du type AX = B avec A = -BA^(-1)B^T et B = c
produit1 = om.produitMat(invA, transB)
Atemp = om.produitMat(om.produitMatNbr(-1, B), produit1)
p = np.linalg.solve( Atemp , c)

x = om.produitMat(invA,om.produitMat(transB, p))

propInvestiePrimal = x

print('propInvestiePrimal', propInvestiePrimal)

##METHODE UZAWA


##RENDEMENT DU PORTEFEUILLE

#Fonction : rendPortefeuille
#Parametres : n : int / nbr d'actions, p : list / proportions investies dans chaque action, r : list / rendement de chaque action
#Retourne : le rendement du portefeuille
def rendPortefeuille(N, p, r):
    rPortefeuille = 0
    for i in N:
        rPortefeuille += p[i] * r[i]
    return rPortefeuille


##OBJECTIFS
#Methode primal-dual

#Methode Uzawa