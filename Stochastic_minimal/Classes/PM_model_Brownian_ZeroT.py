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

class PM_model_Brownian_ZeroT():
    def __init__(self,ws,omega0,gamma,ll,N_R,N_mats,W_mats,integration_limit,t_corr_list,t_list,N_M):

        Gamma = gamma / 2.
        if Gamma >= omega0:
            raise Exception("We should be in the underdamped regime")
        Omega = np.sqrt(omega0**2 - Gamma**2)
        
        N_par = 4
        beta = 'inf'
        mats = matsubara_2(beta,Omega,gamma,ll,N_mats,W_mats,integration_limit,t_corr_list)
        popt, pcov = curve_fit(self.bi_exponential, t_corr_list, np.abs(np.real(mats.mats)),maxfev=1000,bounds=([0 for _ in np.arange(0,N_par)], [100 for _ in np.arange(0,N_par)]))
        freq_array = np.array([popt[1],popt[3]])
        coeff_array = np.array([popt[0],popt[2]])

        Omega_res = Omega
        ll_res = np.sqrt(ll**2 / (2 * Omega))
        Gamma_res = Gamma
        n_res = 0
        Omega_m1 = 0
        Omega_m2 = 0
        ll_m1 = 1j * np.sqrt(abs(coeff_array[0]))
        ll_m2 = 1j * np.sqrt(abs(coeff_array[1]))
        Gamma_m1 = freq_array[0]
        Gamma_m2 = freq_array[1]
        n_m1 = 0
        n_m2 =0

 
        # Operators
        sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
        sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()

        a = tensor(qeye(2),destroy(N_R),qeye(N_M),qeye(N_M))
        b_m1 = tensor(qeye(2),qeye(N_R),destroy(N_M),qeye(N_M))
        b_m2 = tensor(qeye(2),qeye(N_R),qeye(N_M),destroy(N_M))
        sz = tensor(sigmaz,qeye(N_R),qeye(N_M),qeye(N_M))
        sx = tensor(sigmax,qeye(N_R),qeye(N_M),qeye(N_M))

        # Lindblad, observables and initial state
        c_list = []
        obs_list = [sz,a.dag()* a,b_m1.dag() * b_m1, b_m2.dag() * b_m2 ]
        psi0 = tensor(basis(2,1),basis(N_R,0),basis(N_M,0),basis(N_M,0))

        H_S = ws / 2. * sz 
        L = Hamiltonian_single(H_S)
        L = L + create_PM_single(sx,a,Omega_res,ll_res,Gamma_res,n_res)
        L = L + create_PM_single(sx,b_m1,Omega_m1,ll_m1,Gamma_m1,n_m1)
        L = L + create_PM_single(sx,b_m2,Omega_m2,ll_m2,Gamma_m2,n_m2)

        #Dynamics
        args={}
        self.dynamics_full, self.dynamics_a, self.dynamics_m1, self.dynamics_m2  = mesolve(L, psi0, t_list, c_list, obs_list,args=args).expect

        # pickle_out = open("./bath-observables/Classes/Data/mats.dat",'wb')
        # pickle.dump(mats.mats,pickle_out)
        # pickle_out.close()

    def bi_exponential(self,x, a1, b1, a2,b2):
        return a1*np.exp(-b1*abs(x)) + a2*np.exp(-b2*abs(x))