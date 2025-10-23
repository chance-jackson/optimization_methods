import numpy as np
from numpy.linalg import matmul, inv

class optimize:
    def grad(f, x0, delta_x): #compute central finite difference of function
        N = len(x0)
        f_prime = np.zeros((1,N))
        x0_init = x0

        for i in range(N):
            x0_i_init = x0[i]
        
            x0[i] = x0_i_init + delta_x
            f1 = f(x0)

            x0[i] = x0_i_init - delta_x
            f2 = f(x0)

            f_prime[0,i] += (f1 - f2)/(2 * delta_x)
            
            x0 = x0_init

        return f_prime.T #gradient vector

    def hessian(f, x0, delta_x):
        N = len(x0) #length of input vector defines loop variables
        hessian = np.zeros((N,N))
        
        x0_init = x0
        
        for i in range(N):
            
            x0_i_init = x0[i]

            for j in range(N):

                x0_j_init = x0[j] #store initial value in x0[j]

                if i == j:
                    print(x0)
                    x0[i] = x0_i_init + delta_x
                    print(x0)
                    f1 = f(x0)

                    x0[i] = x0_i_init
                    f2 = f(x0)

                    x0[i] = x0_i_init - delta_x
                    f3 = f(x0)

                    hessian[i,j] += (f1 - 2*f2 + f3)/(delta_x**2)
                    
                    x0 = x0_init

                else:
                #cursed and horrible
                    x0[i], x0[j] = x0_i_init + delta_x/2, x0_j_init + delta_x/2 #first function evaluation
                    f1 = f(x0)

                    x0[i], x0[j] = x0_i_init - delta_x/2, x0_j_init + delta_x/2 #continue doing that
                    f2 = f(x0) 

                    x0[i], x0[j] = x0_i_init + delta_x/2, x0_j_init - delta_x/2
                    f3 = f(x0)

                    x0[i], x0[j] = x0_i_init - delta_x/2, x0_j_init - delta_x/2
                    f4 = f(x0)


                    hessian[i,j] += (f1 - f2 - f3 + f4) / (delta_x)**2

                    x0 = x0_init
        
        return hessian
            

def newtons(f, x0, delta_x = 1, tol = 1e-5):
    x_n = x0 + delta_x
    while np.linalg.norm(optimize.grad(f, x0, delta_x)) > tol:

        if np.linalg.det(optimize.hessian(f, x0, delta_x)) == 0: #non-invertible, can occur if curvature around some point is zero or very close to it
            break

        x_n = np.add(x0.reshape(2,1), inv(optimize.hessian(f, x0, delta_x)) @ optimize.grad(f, x0, delta_x))
        x0 = x_n
        
    return x_n
    
