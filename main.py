import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import rosen
from optimize import *
from scipy.differentiate import hessian

x = np.arange(-10,10,0.01)
y = np.arange(-10,10,0.01)

X, Y = np.meshgrid(x,y)

Z = rosen((X,Y))
x = [np.array([-2., -1.]), np.array([-2., 3.]), np.array([2.,-1.]), np.array([2.,3.])]

plt.figure(figsize=(8,6))
for i in x:
    x, x_hist, y_hist, n_iter = newtons(rosen, i, delta_x = 0.001, rate = 1, tol = 1e-10)
    print(n_iter)
#plt.pcolormesh(X, Y, Z, norm = 'log', vmin = 1e-3)
    plt.scatter(x_hist, y_hist, marker = '^', label = f"Starting Point {int(i[0]),int(i[1])}")
#c = plt.colorbar()
#plt.pcolormesh(X, Y, Z, norm = 'log', vmin = 1e-3)
plt.title("Convergence of Newton's Method to Rosenbrock Minima")
plt.scatter(1,1,s=100,label="True Minima")
plt.legend()
plt.ylabel("Y")
plt.xlabel("X")
#plt.savefig("newtons.png",dpi=300)
plt.show()
