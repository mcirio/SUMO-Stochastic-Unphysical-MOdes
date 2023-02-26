import numpy as np

def J_array(g,Omega,ll):
    def f(w):
        return np.pi * g**2 / (2*np.pi*ll) * 1 / np.sqrt(1-(w-Omega)**2/(2*ll)**2)
    return f

    