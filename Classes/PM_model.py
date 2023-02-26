import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )


import numpy as np
from qutip import *
import cmath
from progressbar import progressbar
from correlation_numerical_dynamics_physical_3 import correlation_numerical_dynamics_physical_3


class PM_model():
    def __init__(self,J,W_i,W_f,t_corr_list,beta,n_as_exp,n_s_cut,n_s_noise,H_S,s,N_S,N_PM,psi0_S,t_list,obs_list,integration_limit,interpolation):
        # Parameters
        self.n_s_cut = n_s_cut
        self.n_s_noise = n_s_noise
        self.t_corr_list = t_corr_list
        self.beta = beta
        self.t_list = t_list
        self.n_as_exp = n_as_exp
        self.N_S = N_S
        self.N_PM = N_PM
        self.psi0_S = psi0_S
        self.beta = beta

        # System Hamiltonian and interaction operator as superoperators
        H_S = self.tensor_identities(H_S,n_as_exp)
        s = self.tensor_identities(s,n_as_exp)
        H_xi = self.Hamiltonian_single(s)

        # Creates the PM destruction operators
        self.operators_list = self.generate_operators()

        # Compute the correlation and the PM parameters
        self.C = correlation_numerical_dynamics_physical_3(J,W_i,W_f,t_corr_list,beta,n_as_exp,n_s_cut,n_s_noise,integration_limit,interpolation)
        self.ordered_PM_parameters = self.C.ordered_PM_parameters

        # Compute all necessary to solve the pseudoShroedinger equation
        L = self.generate_L(H_S,s,self.ordered_PM_parameters)
        beta_PM = 'inf' # The PM are at T=0 by construction! This is not the beta of the environment
        psi0 = self.generate_initial_state(beta_PM)
        new_obs_list = []
        for obs in obs_list: 
            new_obs_list.append(self.tensor_identities(obs,n_as_exp))
        c_list = []
        xi_list = self.C.xi_list_s_plus_extra
        self.dynamics, self.sigma_dynamics, self.dynamics_list = self.compute_dynamics(L,H_xi,xi_list,c_list,t_list,psi0,n_s_noise,new_obs_list)
    
    def tensor_identities(self,operator,n):
        if n == 0:
            return operator
        for _ in np.arange(0,n):
             operator = tensor(operator,qeye(self.N_PM))
        return operator
    def generate_operators(self):
        operators_list = []
        for n in np.arange(0,self.n_as_exp):
            operator = qeye(self.N_S)
            operator = self.tensor_identities(operator,self.n_as_exp-1-n)
            operator = tensor(operator,destroy(self.N_PM))
            operator = self.tensor_identities(operator,n)
            operators_list.append(operator)
        return operators_list
    def Hamiltonian_single(self,H):
        return - 1j * (spre(H) - spost(H))
    def Lindblad_single(self,c):
        return (2*spre(c) * spost(c.dag()) - (spre(c.dag() * c) + spost(c.dag() * c)))
    def create_PM_single(self,s,a,Omega,ll,Gamma,n):
        H_PM = ll * s * (a+a.dag()) + Omega * a.dag() * a
        L_PM = self.Hamiltonian_single(H_PM) 
        L_PM = L_PM + Gamma * (n + 1) * self.Lindblad_single(a)  + Gamma * (n) * self.Lindblad_single(a.dag()) 
        return L_PM
    def generate_L(self,H_S,s,ordered_PM_parameters):
        L = self.Hamiltonian_single(H_S)
        for n,a in enumerate(self.operators_list):
            ll = cmath.sqrt(ordered_PM_parameters[n][0])#?#using the real case
            Omega = ordered_PM_parameters[n][1]
            #if Omega < 0: Omega = - Omega
            Gamma = ordered_PM_parameters[n][2]#?#using the real case
            n_th = 0
            L = L + self.create_PM_single(s,a,Omega,ll,Gamma,n_th)
        return L
    def compute_dynamics(self,L,H_xi,xi_list,c_list,t_list,psi0,n_noise,obs_list):
        sigma = 1
        dynamics_average = 0
        dynamics_list = []    
        k = 0
        print("Averaging the dynamics")
        for k in progressbar(np.arange(0,n_noise)):
            xi = xi_list[k]
            L_xi = [H_xi,xi] 
            args = {}
            options = Options(num_cpus=4, atol=1e-15,nsteps = 100000000)
            dynamics_list.append(mesolve([L,L_xi], psi0, t_list, c_list, obs_list,args=args,options=options).expect[0])
            #dynamics_list.append(mesolve(L, psi0, t_list, c_list, obs_list,args=args,options=options).expect[0])
            dynamics_average += dynamics_list[-1]
        dynamics_average = dynamics_average / n_noise
        sigma = 0 * np.array(dynamics_average)
        # for dynamics in dynamics_list:
        #     sigma += (dynamics - dynamics_average)**2
        # sigma = np.sqrt(sigma / len(dynamics_list))
        for dynamics in dynamics_list:
            sigma += np.array(dynamics)**2
        sigma = np.sqrt(sigma / len(dynamics_list) - np.array(dynamics_average)**2) / np.sqrt(self.n_s_noise)
        return dynamics_average, sigma, dynamics_list
    def generate_initial_state(self,beta_PM):
        psi0 = self.psi0_S
        for n in np.arange(0,self.n_as_exp):
            psi0_PM = self.create_PM_state(beta_PM,self.N_PM)
            psi0 = tensor(psi0,psi0_PM)
        return psi0   
    def create_PM_state(self,beta,N_PM):
        if beta == 'inf':
            psi = basis(N_PM,0) * basis(N_PM,0).dag()
            return psi
        psi = (-beta * destroy(N_PM).dag() * destroy(N_PM)).expm()
        psi = psi / psi.tr()
        return psi

    def save(self,dict):
        dict['n_s_cut'] = self.n_s_cut
        dict['n_s_noise'] = self.n_s_noise
        dict['beta'] = self.beta
        dict['t_corr_list'] = self.t_corr_list
        dict['t_list'] = self.t_list

        dict['C_s'] = self.C.C_s
        dict['C_as'] = self.C.C_as
        dict['C_s_extra'] = self.C.C_s_extra
        dict['C_as_fit'] = self.C.C_as_fit
        dict['check'] = self.C.check
        dict['check_fit'] = self.C.check_fit

        dict['C_as_minus_extra'] = self.C.C_as_minus_extra
        dict['C_as_minus_extra_fit'] = self.C.C_as_minus_extra_fit
        dict['C_s_plus_extra'] = self.C.C_s_plus_extra
        dict['C_s_plus_extra_reconstructed'] = self.C.C_s_plus_extra_reconstructed
        dict['C_s_plus_extra_stochastic'] = self.C.C_s_plus_extra_stochastic
        dict['dynamics'] = self.dynamics
        dict['dynamics_list'] = self.dynamics_list
        dict['ordered_PM_parameters'] = self.ordered_PM_parameters

        dict['expected_error'] = self.C.C_s_plus_extra_class.expected_error
        dict['sigma_dynamics'] = self.sigma_dynamics
        dict['dynamics_list'] = self.dynamics_list
        dict['xi_list'] = self.C.C_s_plus_extra_class.xi_list

        return dict
