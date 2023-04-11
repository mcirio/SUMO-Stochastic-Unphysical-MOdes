
"""
This module constructs models and solves the dynamics of the orignal system-
bath problem in the expanded pseudo-mode space.
"""

from qutip import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import numpy as np
from sumo.Functions.compute_correlations import compute_correlations
from sumo.Functions.fit import fit 
from sumo.Functions.spectral_decomposition import spectral_decomposition
from sumo.Functions.generate_fields import generate_fields
from sumo.Functions.generate_PM_model import generate_PM_model
from sumo.Functions.average_dynamics import average_dynamics,average_dynamics_parallel
from sumo.Functions.quantum_functions import *
from sumo.Classes.bath_correlations import matsubara_2

__all__ = [
    "Stochastic_model",
]

class Stochastic_model():
    """
    Constructs a stochastic version of the pseudomode-model specifically for the
    zero-temeperature under-damped brownian motion spectral density
    
    Parameters
    ----------
    corr_params : 
    
    system_params : 
        
    PM_params :
    
    stoch_params : 
    
    
    Attributes
    ----------
    
    """
    def __init__(self,corr_params, system_params,PM_params,stoch_params):

        J = corr_params['J']
        beta = corr_params['beta'] 
        W_i = corr_params['W_i']
        W_f = corr_params['W_f'] 
        integration_limit = corr_params['integration_limit']
        t_corr_list = corr_params['t_corr_list']

        N_S = system_params['N_S']
        H_S = system_params['H_S']
        s = system_params['s']
        psi0_S = system_params['psi0_S']
        obs_list = system_params['obs_list']
        t_list = system_params['t_list']

        N_PM = PM_params['N_PM'] 
        n_as_exp = PM_params['n_as_exp'] 

        n_cut = stoch_params['n_cut'] 
        n_noise = stoch_params['n_noise'] 

        self.C_s, self.C_as = compute_correlations(J,beta,W_i,W_f,integration_limit,t_corr_list)
        self.ordered_PM_parameters, self.C_as_fit, self.C_s_extra = fit(t_corr_list,self.C_as,n_as_exp)

        self.C_s_plus_extra = [x[0] + x[1] for x in zip(self.C_s,self.C_s_extra)]
        self.C_as_minus_extra = [x[0] - x[1] for x in zip(self.C_as,self.C_s_extra)]

        self.coeff_list, self.C_s_reconstructed = spectral_decomposition(t_corr_list,self.C_s_plus_extra,n_cut)
        self.xi_interpolated_list, self.C_s_reconstructed_stochastic = generate_fields(t_corr_list,self.coeff_list,n_cut,n_noise)

        self.H_S, self.s, self.H_xi, self.L, self.psi0, self.new_obs_list, self.c_list = generate_PM_model(H_S,psi0_S,s,self.ordered_PM_parameters,obs_list,N_S,N_PM)
        self.dynamics_average, self.sigma, self.dynamics_list = average_dynamics_parallel(self.L,self.H_xi,self.xi_interpolated_list,self.c_list,t_list,self.psi0,n_noise,self.new_obs_list)
    

class PM_model_Brownian_ZeroT():
    """
    Constructs a fully quantum version of the pseudomode-model specifically for the
    zero-temeperature under-damped brownian motion spectral density
    
    This needs a complete rewrite
    

    
    
    Attributes
    ----------
    
    """
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
        
        
        

class PM_model_Brownian_ZeroT_noMats():
    """
    Constructs a fully quantum version of the pseudomode-model specifically for the
    zero-temeperature under-damped brownian motion spectral density with no matsubara terms.
    
    This needs a complete rewrite
    

    
    
    Attributes
    ----------
    
    """
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