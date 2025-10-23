import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import rosen
from optimize import *
from scipy.differentiate import hessian

x = np.arange(-2,2,0.01)
y = np.arange(-1,3,0.01)

X, Y = np.meshgrid(x,y)

Z = rosen((X,Y))
x = np.array([0.5,0.5])
print(optimize.hessian(rosen, x, 1))
print(hessian(rosen,x))

print(newtons(rosen, x))
#plt.pcolormesh(X, Y, Z, norm = 'log', vmin = 1e-3)
#c = plt.colorbar()
#plt.show()
