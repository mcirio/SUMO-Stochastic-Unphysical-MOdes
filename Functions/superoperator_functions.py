from qutip import *

def Hamiltonian_single(H):
    return - 1j * (spre(H) - spost(H))
def Lindblad_single(c):
    return (2*spre(c) * spost(c.dag()) - (spre(c.dag() * c) + spost(c.dag() * c)))
def create_PM_single(s,a,Omega,ll,Gamma,n):
    H_PM = ll * s * (a+a.dag()) + Omega * a.dag() * a
    L_PM = Hamiltonian_single(H_PM) 
    L_PM = L_PM + Gamma * (n + 1) * Lindblad_single(a)  + Gamma * (n) * Lindblad_single(a.dag()) 
    return L_PM
def create_PM_state(beta,Omega,N_PM):
    if (beta == 'inf') & (Omega != 0):
        psi = basis(N_PM,0) * basis(N_PM,0).dag()
        return psi
    if (beta == 'inf') & (Omega == 0):
        beta = 0
    psi = (-beta * Omega * destroy(N_PM).dag() * destroy(N_PM)).expm()
    psi = psi / psi.tr()
    return psi

