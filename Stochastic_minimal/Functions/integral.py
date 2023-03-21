import numpy as np
from scipy.integrate import quad

def integral(function,*args,**kwargs):
    if 'x_i' not in kwargs:
          raise Exception('Missing x_i, {} instead'.format(kwargs.keys()))
    if 'x_f' not in kwargs:
          raise Exception('Missing x_f, {} instead'.format(kwargs.keys()))
    x_i = kwargs['x_i']
    x_f = kwargs['x_f']
    limit = kwargs['limit']
    def function_real(x,*args):
        return np.real(function(x,*args))
    def function_imag(x,*args):
        return np.imag(function(x,*args))
        
    quadv = np.vectorize(quad)

    return quadv(function_real, x_i, x_f,args=args,limit=limit)[0] + 1j * quadv(function_imag, x_i, x_f,args=args,limit=limit)[0]