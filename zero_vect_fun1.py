from scipy import optimize
import numpy as np


R=1.025 # risque
r0=1.001
r1=1.050
r2=1.008
r=[r0,r1,r2]


def fun(x):
    return [x[0]*np.cov(x[0],x[0]) + x[1]*np.cov(x[0],x[1]) + x[2]*np.cov(x[0],x[2]),
            x[0]*np.cov(x[1],x[0]) + x[1]*np.cov(x[1],x[1]) + x[2]*np.cov(x[1],x[2]),
            x[0]*np.cov(x[2],x[0]) + x[1]*np.cov(x[2],x[1]) + x[2]*np.cov(x[2],x[2]),
            x[0]  + x[1] + x[2] - 1.00,       #E1(x)
            r0*x[0]+ r1*x[0] +r2*x[1] - R     #E2(x)
            ]

#
def jacobian(x):
    return np.array([[2 , -1 ,  1 , 1, x[0]*(x[0]-4)],
                     [-1 , 0 ,  1 , 1, 1],
                     [ 1 , 1 ,  0 , 1, 1],
                     [ 1 , 1 ,  1 , 0, 0],
                     [ r0 , r1 ,  r2 , 0, 0]                     
                     ])        
        
sol = optimize.root(fun, [10,10,110,0,0], jac=jacobian)
print('with jacobian=',sol.x)
x=sol.x
print('E1(x) constraint=',x[0]  + x[1] + x[2] - 1)
print('E2(x) constraint=',r0*x[0]+ r1*x[0] +r2*x[1] - R)
print('---------------------------------------------------------')
sol = optimize.root(fun, [100,110,110,0,0], method='broyden1')
print('broyden1 =',sol.x)
x=sol.x
print('E1(x) constraint=',x[0]  + x[1] + x[2] - 1)
print('E2(x) constraint=',r0*x[0]+ r1*x[0] +r2*x[1] - R)
print('---------------------------------------------------------')
sol = optimize.root(fun, [10000,100,1,0,0], method='broyden2')
print('broyden2 (less accurate) =',sol.x)
x=sol.x
print('E1(x) constraint=',x[0]  + x[1] + x[2] - 1)
print('E2(x) constraint=',r0*x[0]+ r1*x[0] +r2*x[1] - R)
print('---------------------------------------------------------')
#J=-x[0]*x[1]-x[0]*x[2]-x[1]*x[2]  #11
#J=x[0]**2-4*x[0]-x[0]*x[1]+x[0]*x[2]+x[1]*x[2]  #12

Jopt=-2
print('J(x)=',J)
print('sensibility of J wrt E',(J-Jopt)/(0.001),x[3])

#parallelisme entre gradJ et grad E a l'optimum ?
# < grad J , grad E > = +/- sqrt(< gradJ , grad J > < gradE , grad E >) ?
# ou bien on verifie que le produit vectoriel est proche de zero.
#    (-2, -2, -2)(1, 1, 1) = -6  vs. sqrt(12*3)=6 ok  pour ex.11

#idem ex.12. et avec 2 contraintes













#def fun(x):
#    return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
#            0.5 * (x[1] - x[0])**3 + x[1]]
#
#
#def jacobian(x):
#    return np.array([[1 + 1.5 * (x[0] - x[1])**2,
#                      -1.5 * (x[0] - x[1])**2],
#                     [-1.5 * (x[1] - x[0])**2,
#                      1 + 1.5 * (x[1] - x[0])**2]])