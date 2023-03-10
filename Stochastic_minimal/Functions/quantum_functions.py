import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np
from qutip import *
import cmath

def tensor_identities(operator,n,space_dim):
    if n == 0:
        return operator
    for _ in np.arange(0,n):
            operator = tensor(operator,qeye(space_dim))
    return operator
def generate_operators(N_modes,space_system_dim,space_modes_dim):
    operators_list = []
    for n in np.arange(0,N_modes):
        operator = qeye(space_system_dim)
        operator = tensor_identities(operator,N_modes-1-n,space_modes_dim)
        operator = tensor(operator,destroy(space_modes_dim))
        operator = tensor_identities(operator,n,space_modes_dim)
        operators_list.append(operator)
    return operators_list
def Hamiltonian_single(H):
    return - 1j * (spre(H) - spost(H))
def Lindblad_single(c):
    return (2*spre(c) * spost(c.dag()) - (spre(c.dag() * c) + spost(c.dag() * c)))
def create_PM_single(s,a,Omega,ll,Gamma,n):
    H_PM = ll * s * (a+a.dag()) + Omega * a.dag() * a
    L_PM = Hamiltonian_single(H_PM) 
    L_PM = L_PM + Gamma * (n + 1) * Lindblad_single(a)  + Gamma * (n) * Lindblad_single(a.dag()) 
    return L_PM
def generate_initial_state(psi0_S,beta,N_modes,space_modes_dim):
    psi0 = psi0_S
    for n in np.arange(0,N_modes):
        psi0_PM = create_PM_state(beta,space_modes_dim)
        psi0 = tensor(psi0,psi0_PM)
    return psi0   
def create_PM_state(beta,space_modes_dim):
    if beta == 'inf':
        psi = basis(space_modes_dim,0) * basis(space_modes_dim,0).dag()
        return psi
    psi = (-beta * destroy(space_modes_dim).dag() * destroy(space_modes_dim)).expm()
    psi = psi / psi.tr()
    return psi
def generate_L(H_S,s,ordered_PM_parameters,operators_list):
    L = Hamiltonian_single(H_S)
    for n,a in enumerate(operators_list):
        ll = cmath.sqrt(ordered_PM_parameters[n][0])
        Omega = ordered_PM_parameters[n][1]
        Gamma = ordered_PM_parameters[n][2]
        n_th = 0
        L = L + create_PM_single(s,a,Omega,ll,Gamma,n_th)
    return L