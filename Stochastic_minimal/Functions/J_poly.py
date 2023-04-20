def J_poly(vec_p,vec_w):
    a = vec_w[0]
    b = vec_w[1]
    c = vec_w[2]
    d = vec_w[3]
    def f(w):
        return (vec_p[1] * w)/(w-a)/(w-b)/(w-c)/(w-d)
    return f