import numpy as np

def J_transmissionLine(Dx,c,ll):
    def f(w):
        return ll**2 / (2 * np.pi * c) * w / np.sqrt(1-w**2 * Dx**2/(4*c**2) )
    return f

    