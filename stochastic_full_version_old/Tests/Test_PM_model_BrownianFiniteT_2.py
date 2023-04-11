import os
import sys
import pickle

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Classes' )
sys.path.append( mymodule_dir )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.optimize import curve_fit
import cmath
from qutip.ui.progressbar import BaseProgressBar
from J_poly import J_poly
from PM_model import PM_model
from regularize_poles import regularize_poles
from matsubara import matsubara_2
from utility_functions import coth
from superoperator_functions import Hamiltonian_single,Lindblad_single,create_PM_state,create_PM_single


# Frequency unit
w0 = 2 * np.pi 
# Cut offs
W_mats = 10 * w0
N_mats = 1000
N_corr = 500 #500#500#1000
N_cut = 1000
N_M = 3
N_M2 = 3
N_R = 5
W_free = 100 * w0
W_cut = 10 * w0
n_cut = 500
mats_cut = 500
n_noise = 100
n_as_exp = 1
n_s_cut = 500
n_s_noise = 500  #50#100#500
W_i = 0
W_f = 10 * w0
integration_limit = 500
# Parameters
w_tilde = W_mats
gamma = .1 * w0
Gamma = gamma / 2.
Omega = np.sqrt(w0**2 - Gamma**2)
ll = .2 * w0 #* np.sqrt(2*Omega)
T = 20. * 2 * np.pi / w0
time_stamp = 0
norm = 1.
vec_p = [0,2 * Gamma * ll**2]
vec_w = [Omega + 1j*Gamma]
vec_w = regularize_poles(vec_w)
J = J_poly(vec_p,vec_w)
beta = 2. / w0
Omega_S = w0
#th = -1
th = (np.exp(-beta*Omega_S)-1) / (np.exp(-beta*Omega_S)+1)
# Lists
t_list = np.linspace(0,T,500)
t_list_dynamics = t_list
t_list_scaled = [x * w0 / (2*np.pi) for x in t_list] #Time in units of 2\pi/w0
t_list_corr = np.linspace(-T,T,2*N_corr+1)
t_corr_list = np.linspace(-T,T,2*N_corr+1)
t_list_interp = np.linspace(-1.1*T,1.1*T,5000)
w_list = np.linspace(0,3*w0,100)
W_list = np.linspace(W_i,W_f,200)
## Operators 
N_S = 2
N_PM = 5
sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()
H_S = Omega_S / 2. * sigmaz
s = sigmax
psi0_S = basis(2,1) * basis(2,1).dag()
obs_list = [sigmaz]

###################################################
interpolation = 'no'
#PM = PM_model_analytical(Omega,ll,gamma,t_corr_list,beta,n_s_cut,n_s_noise,H_S,s,N_S,N_PM,psi0_S,t_list,obs_list,integration_limit,N_mats,W_mats)
PM = PM_model(J,W_i,W_f,t_corr_list,beta,n_as_exp,n_s_cut,n_s_noise,H_S,s,N_S,N_PM,psi0_S,t_list,obs_list,integration_limit,interpolation)
###################################################
## PM model
#Fit
N_par = 4
def bi_exponential(x, a1, b1, a2,b2):
     return a1*np.exp(-b1*abs(x)) + a2*np.exp(-b2*abs(x))
mats = matsubara_2(beta,Omega,gamma,ll,N_mats,W_mats,integration_limit,t_corr_list)
popt, pcov = curve_fit(bi_exponential, t_corr_list, np.abs(np.real(mats.mats)),maxfev=1000,bounds=([0 for _ in np.arange(0,N_par)], [100 for _ in np.arange(0,N_par)]))
freq_array = np.array([popt[1],popt[3]])
coeff_array = np.array([popt[0],popt[2]])

#print(freq_array)
#print(coeff_array)
R = np.real(coth(beta,(Omega+1j*Gamma)/2.))
I = np.imag(coth(beta,(Omega+1j*Gamma)/2.))


Omega_res = Omega
ll_res = np.sqrt(ll**2 / (2 * Omega))
Gamma_res = Gamma
n_res = (R - 1) / 2.
print('n_res=', n_res)
beta_res = np.log(1+1/n_res) / Omega_res
Omega_m1 = 0
Omega_m2 = 0
ll_m1 = 1j * np.sqrt(abs(coeff_array[0]))
ll_m2 = 1j * np.sqrt(abs(coeff_array[1]))
Gamma_m1 = freq_array[0]
Gamma_m2 = freq_array[1]
n_m1 = 0
n_m2 = 0
beta_mats = 'inf'

#w1 = Omega + 1j * Gamma
#w2 = -Omega + 1j * Gamma
Omega_res1 = 0
Omega_res2 = 0
ll_res1 = cmath.sqrt(ll**2 / (4 * Omega) * I)
ll_res2 = cmath.sqrt(-ll**2 / (4 * Omega) * I)
Gamma_res1 = -1j * Omega + Gamma
Gamma_res2 = 1j * Omega + Gamma
n_res1 = 0
n_res2 = 0
beta_res1 = 'inf'
beta_res2 = 'inf'

# Operators
sigmaz = basis(2,1) * basis(2,1).dag() - basis(2,0) * basis(2,0).dag()
sigmax = basis(2,1) * basis(2,0).dag() + basis(2,0) * basis(2,1).dag()

a = tensor(qeye(2),destroy(N_R),qeye(N_M),qeye(N_M),qeye(N_M2),qeye(N_M2))
b_m1 = tensor(qeye(2),qeye(N_R),destroy(N_M),qeye(N_M),qeye(N_M2),qeye(N_M2))
b_m2 = tensor(qeye(2),qeye(N_R),qeye(N_M),destroy(N_M),qeye(N_M2),qeye(N_M2))
b_res1 = tensor(qeye(2),qeye(N_R),qeye(N_M),qeye(N_M),destroy(N_M2),qeye(N_M2))
b_res2 = tensor(qeye(2),qeye(N_R),qeye(N_M),qeye(N_M),qeye(N_M2),destroy(N_M2))
sz = tensor(sigmaz,qeye(N_R),qeye(N_M),qeye(N_M),qeye(N_M2),qeye(N_M2))
sx = tensor(sigmax,qeye(N_R),qeye(N_M),qeye(N_M),qeye(N_M2),qeye(N_M2))

c_list = []
obs_list = [sz]
beta_PM = 'inf' #PM are at zero T!
psi0 = tensor(basis(2,1)*basis(2,1).dag(),create_PM_state(beta_res,Omega_res,N_R),create_PM_state(beta_mats,Omega_m1,N_M),create_PM_state(beta_mats,Omega_m2,N_M),create_PM_state(beta_res1,Omega_res2,N_M2),create_PM_state(beta_res2,Omega_res2,N_M2))

H_S = Omega_S / 2. * sz 
L = Hamiltonian_single(H_S)
L = L + create_PM_single(sx,a,Omega_res,ll_res,Gamma_res,n_res)
L = L + create_PM_single(sx,b_m1,Omega_m1,ll_m1,Gamma_m1,n_m1)
L = L + create_PM_single(sx,b_m2,Omega_m2,ll_m2,Gamma_m2,n_m2)
L = L + create_PM_single(sx,b_res1,Omega_res1,ll_res1,Gamma_res1,n_res1)
L = L + create_PM_single(sx,b_res2,Omega_res2,ll_res2,Gamma_res2,n_res2)

#Matusbara Correlations
#mats = [Matsubara_zeroT(Omega,gamma,ll,t,W_mats) for t in t_list]

#Dynamics
args={}
dynamics_full = mesolve(L, psi0, t_list, c_list, obs_list,progress_bar=True,args=args).expect[0]
print('Finished PM model')
##################################################################
pickle_out = open("./PM_3/Tests/Data/Test_PM_model_BrownianFiniteT.dat",'wb')
dict = {}
dict_2 = {}
#dict = PM.save(dict)
#dict['dynamics_full'] = dynamics_full
dict = PM.save(dict_2)
dict['dynamics_full'] = dynamics_full
#dict['dynamics_no_Matsubara'] = dynamics_no_Matsubara
dict['Omega_S'] = Omega_S
dict['ll'] = ll
dict['Omega'] = Omega
dict['Gamma'] = Gamma
to_save = [dict,th]
pickle.dump(to_save,pickle_out)
pickle_out.close()
###############################################################
fig, axs = plt.subplots(4, 1,figsize=(17,12))
axs[0].set_title('Symmetric + extra')
#axs[0].plot(t_corr_list,np.real(PM.C_s_list),'b',label='analytical (beta={})'.format(beta))
#axs[0].plot(t_corr_list,np.real(PM.C_s_reconstructed),'r--',label='reconstructed')
#axs[0].plot(t_corr_list,np.real(PM.C_s_stochastic),'g--',label='stochastic')
axs[0].legend()

axs[1].set_title('Symmetric + extra (two methods)')
axs[1].plot(t_corr_list,PM.C.C_s_plus_extra,'b')
axs[1].plot(t_corr_list,PM.C.C_s_plus_extra_stochastic,'g--')
axs[1].plot(t_corr_list,PM.C.C_s_plus_extra_reconstructed,'r--')

axs[2].set_title('Antisymmetric - extra (two methods)')
axs[2].plot(t_corr_list,PM.C.C_as_minus_extra,'b')
axs[2].plot(t_corr_list,PM.C.C_as_fit,'b')
#axs[2].plot(t_corr_list,[ll**2/(2*Omega)*np.exp(-1j*Omega*t-Gamma*abs(t)) for t in t_corr_list],'r--')

axs[3].set_title('Dynamics')
axs[3].plot(t_list,dynamics_full,'b',label='full PM')
#axs[3].plot(t_list,PM.dynamics,'g--',label='stochastic')
#axs[3].plot(t_list,PM.dynamics_average,'r--',label='stochastic')
axs[3].plot(t_list,[th for _ in t_list],color='darkgrey',linestyle='--')
axs[3].legend()
#axs[1].set_ylim(-1.01,-.93)

plt.show()