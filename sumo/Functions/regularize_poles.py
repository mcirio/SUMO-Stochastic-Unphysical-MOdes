import numpy as np

def regularize_poles(vec_w):
    vec_w_add = []
    for w in vec_w:
        if np.real(w) == 0:
            vec_w_add.append(np.conj(w))
        else:
            vec_w_add.append(np.conj(w))
            vec_w_add.append(-w)
            vec_w_add.append(np.conj(-w))
    vec_w = vec_w + vec_w_add 
    return vec_w
