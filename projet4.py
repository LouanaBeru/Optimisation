##README
#This code uses operations_Matricielles available on our website
#Please make sure to save those two codes in the same directory before running projet4

###############################################
##MODULES
import matplotlib.pyplot as plt
import numpy as np
import operations_Matricielles as om

###############################################
##EXCTRACTION DES DONNEES DE TEMPERATURE.TXT
temperature = open ('temperature.txt', 'r') #va chercher le dossier, D:/Users/Chapati/Documents/Fac/S6/Optimisation/VSCODE_OPTI/semaine 4/ si c'etait pas au même endroit
temperatureTableau = temperature.read().split('\n') #renvoie un tableau dont chaque element est une ligne du fichier
temperature.close() #ferme le fichier

donnee = []
for i in temperatureTableau: #a chaque boucle, i vaut la ... ligne du tableau
    donneeSplit = i.split(' ')#recupère toutes les donnees du tableau separees par des espaces
    donneeChiffre = []
    for j in donneeSplit:
        if j != '':
            donneeChiffre.append(float(j)) #on prend que les valeurs non-nulles et au passage on les converti en chiffre
    donnee.append(donneeChiffre)
# print(donnee)
# print(donnee[0][0])

###############################################
##CREATION DE VECTEURS DONNEES

dates = []
for i in donnee:
    dates.append(i[0])
dates = np.asarray(dates)

###premiere colonne

temperature1 = []
for i in donnee:
    temperature1.append(i[1])
temperature1 = np.asarray(temperature1)

# plt.figure(1)
# plt.scatter(dates, temperature1, c='black')

###deuxieme colonne

temperature2 = []
for i in donnee:
    temperature2.append(i[2])
temperature2 = np.asarray(temperature2)

# plt.figure(2)
# plt.scatter(dates, temperature2, c='black')
# plt.show()

###############################################
##NORMALISATION

temperatureMin1 = np.amin(temperature1)
temperatureMin2 = np.amin(temperature2)
temperatureMax1 = np.amax(temperature1)
temperatureMax2 = np.amax(temperature2)
dateMin = np.amin(dates)
dateMax = np.amax(dates)

for k in range(len(temperature1)):
    temperature1[k] = (temperature1[k] - temperatureMin1) / (temperatureMax1 - temperatureMin1 )

for k in range(len(temperature2)):
    temperature2[k] = (temperature2[k] - temperatureMin2) / (temperatureMax2 - temperatureMin2 )

for k in range(len(dates)):
    dates[k] = (dates[k] - dateMin) / (dateMax - dateMin)

###############################################
##MOINDRE CARRE DE DEGRE 1
###DEFINITION MATRICES

X = []
for i in range(len(dates)):
    X.append([1, dates[i]])
X = np.asarray(X)

Y1 = []
for i in range(len(temperature1)):
    Y1.append(temperature1[i])
#Y1 = np.asarray(Y1)

Y2 = []
for i in range(len(temperature2)):
    Y2.append(temperature2[i])
#Y2 = np.asarray(Y2)

#On va résoudre un système du type AX = B

A = om.produitMat(om.transpose(X), X)
B1 = om.produitMat(om.transpose(X), Y1)
B2 = om.produitMat(om.transpose(X), Y2)

###RECHERCHE DES COEFFICIENTS

X1 = np.linalg.solve(A, B1)

X2 = np.linalg.solve(A, B2)

###CONSTRUCTION POLYNOME D APPROXIMATION

Droite1 = []
Droite2 = []

for i in dates:
    Droite1.append(X1[1]*i + X1[0])
    Droite2.append(X2[1]*i + X2[0])


###AFFICHAGE INTERMEDIAIRE
plt.figure(1)
plt.scatter(dates, temperature1, c='black')
plt.plot(dates, Droite1, c = 'red' )

plt.figure(2)
plt.scatter(dates, temperature2, c='black')
plt.plot(dates, Droite2, c = 'red' )

###############################################
##MOINDRE CARRE DE DEGRE 2
###DEFINITION MATRICES

X = []
for i in range(len(dates)):
    X.append([1, dates[i], dates[i]**2 ])
X = np.asarray(X)

A = om.produitMat(om.transpose(X), X)
B1 = om.produitMat(om.transpose(X), Y1)
B2 = om.produitMat(om.transpose(X), Y2)

###RECHERCHE DES COEFFICIENTS
X1 = np.linalg.solve(A, B1)
X2 = np.linalg.solve(A, B2)

###CONSTRUCTION POLYNOME D APPROXIMATION
Polynom1 = []
Polynom2 = []

for i in dates:
    Polynom1.append(X1[2] * i**2 + X1[1]*i + X1[0])
    Polynom2.append(X2[2] * i**2 + X2[1]*i + X2[0])

###AFFICHAGE FINAL
plt.figure(1)
plt.plot(dates, Polynom1, c = 'green' )

plt.figure(2)
plt.plot(dates, Polynom2, c = 'green' )

plt.show()