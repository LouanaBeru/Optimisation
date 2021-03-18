##VARIATIONS SUR 1 AN TOUT LES ~18 DU MOIS, VALEUR DE L ACTION A LA CLOTURE

###L OREAL
O = [228.6, 256.8, 253.6, 278.7, 290.7, 281.4, 280.8, 295, 317.4, 306.6, 298.7, 319.1, 326.4]

###RENAULT
R = [16.478, 17.028, 18.998, 22.335, 24.14, 24.685, 23.585, 24.165, 31.48, 37.125, 34.815, 40.135, 39.975]

###MICHELIN
M = [75.06, 90.48, 88.6, 91.5, 94.62, 96.8, 96.14, 95.1, 108.6, 109.6, 109.35, 117.95, 127.05]



def moy(x):
    m=0
    for i in range(len(x)):
        m+=x[i]
    return m/len(x)


def cov(s,t):
    c=0
    S=moy(s)
    T=moy(t)
    for i in range(len(s)):
        c+=(s[i]-S)*(t[i]-T)
    return c/len(s)

r=[.,.,.]
R0=.


##METHODE zero_vect_fun
#J(x) = 1/2 * < Ax ; x >
#E1 = x[0]+x[1]+x[2]-1
#E2 = x[0]*r[0] +x[1]*r[1] +x[2]*r[2]-R
#x = (proportion investie dans l'Oreal, prop. invest dans Renault, prop. invest dans Michelin)
#On cherche le x qui minimise J
def fun(x):
    return [(x[0]*cov(O,O) + x[1]*cov(O,R) + x[2]*cov(O,M))/2 + x[3] + x[4],
            (x[0]*cov(R,O) + x[1]*cov(R,R) + x[2]*cov(R,M))/2 + x[3] + x[4],
            (x[0]*cov(M,O) + x[1]*cov(M,R) + x[2]*cov(M,M))/2 + x[3] + x[4],
            x[0]+x[1]+x[2]-1,
            x[0]*r[0] +x[1]*r[1] +x[2]*r[2]-R0
            ]

def jacobian(x):
    returnnp.array([[ cov(O,O) , cov(O,R) ,  cov(O,M) , 1, 1],
                     [cov(R,O) , cov(R,R) ,  cov(R,M) , 1, 1],
                     [cov(M,O) , cov(M,R) ,  cov(M,M) , 1, 1],
                     [ 1 , 1 ,  1 , 0, 0],
                     [ r[0] , r[1] ,  r[2] , 0, 0]                     
                     ])
                     
##OBJECTIFS
#r = R0 = le revenu qu'on souhaite obtenir
#ri = r[i-1] = le revenu du i-eme actif
# -> Vérifier les matrices et appliquer zero_vect_fun

#Methode primal-dual

#Methode Uzawa