import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..','Functions' )
sys.path.append( mymodule_dir )
import pickle

import numpy as np
from scipy.interpolate import interp1d
# from integral import integral
from progressbar import progressbar
from matsubara_2 import matsubara_2
from scipy.optimize import curve_fit
from qutip import *
from quantum_functions import *

class PM_model_Brownian_ZeroT_noMats():
    def __init__(self,ws,omega0,gamma,ll,t_list,N_R):

        Gamma = gamma / 2.
        if Gamma >= omega0:
            raise Exception("We should be in the underdamped regime")
        Omega = np.sqrt(omega0**2 - Gamma**2)
        
        Omega_res = Omega
        ll_res = np.sqrt(ll**2 / (2 * Omega))
        Gamma_res = Gamma
        n_res = 0
 
        # Operators
        sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
        sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()

        a = tensor(qeye(2),destroy(N_R))
        sz = tensor(sigmaz,qeye(N_R))
        sx = tensor(sigmax,qeye(N_R))

        # Lindblad, observables and initial state
        c_list = []
        obs_list = [sz]
        psi0 = tensor(basis(2,1),basis(N_R,0))

        H_S = ws / 2. * sz 
        L = Hamiltonian_single(H_S)
        L = L + create_PM_single(sx,a,Omega_res,ll_res,Gamma_res,n_res)

        #Dynamics
        args={}
        self.dynamics_full = mesolve(L, psi0, t_list, c_list, obs_list,args=args).expect[0]

    def bi_exponential(self,x, a1, b1, a2,b2):
        return a1*np.exp(-b1*abs(x)) + a2*np.exp(-b2*abs(x))