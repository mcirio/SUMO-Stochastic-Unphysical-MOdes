import numpy as np
def J_ohmic(ll,wc):
    def f(w):
        return 1 / np.pi * ll * w * np.exp(-w/wc)
    return f