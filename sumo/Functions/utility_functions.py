import numpy as np

def coth(beta,x):
    if x == 0:
        raise Exception('argument should not be zero')
    if beta == 'inf':
        return sg(np.real(x))
    return (np.exp(beta*x) + np.exp(-beta*x)) / (np.exp(beta*x) - np.exp(-beta*x))
def sg(t):
    if t > 0:
        return 1
    return -1
def theta(t):
    if t < 0:
        return 0
    return 1