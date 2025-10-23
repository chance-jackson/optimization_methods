from optimize import *
import numpy as np
from scipy.differentiate import hessian
from numpy.linalg import norm
from scipy.optimize import rosen

def f(x):
    return x[0]**3 + x[1]**3

k = np.array([1,1])

print(optimize.hessian(rosen, k, 2))
print(hessian(rosen,k))
