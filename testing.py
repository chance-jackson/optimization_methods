from optimize import *
import numpy as np
from scipy.differentiate import hessian
from numpy.linalg import norm


f = lambda x: x[0]**3 + x[1]**3

x = np.array([1,1])

print(optimize.hessian(f, x, 0.5))
print(hessian(f,x))
