def J_poly(vec_p,vec_w):
    def f(w):
        res = 0
        for k,p_k in enumerate(vec_p):
            res += p_k * w**k
        for w_k in vec_w:
            res = res / (w - w_k)
        return res
    return f