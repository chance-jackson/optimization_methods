import numpy as np

class optimize:
    def grad(f, x0, delta_x): #compute central finite difference of function

        f_prime = (f(x0+delta_x) - f(x0-delta_x))/(2 * delta_x)

        return f_prime #gradient vector

    def hessian(f, x0, delta_x):
        N = len(x0) #length of input vector defines loop variables
        hessian = np.zeros(N,N)
        
        for i,j in range(N):
            hessian[i,j] += (f(x0[i] + delta_x[i], x0[j] + delta_x[j]) 
            
                


        

#    def newtons(f, delta_x):

