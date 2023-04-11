import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..\..', 'Classes' )
sys.path.append( mymodule_dir )
mymodule_dir = os.path.join( script_dir, '..\..', 'Functions' )
sys.path.append( mymodule_dir )

from regularize_poles import regularize_poles
from J_poly import J_poly
from correlation_numerical import correlation_numerical
from correlation_single_brownian import correlation_single_brownian

import numpy as np
import matplotlib.pyplot as plt
import pickle
import cmath


# Units
w = 2 * np.pi
# Cuts 
w_cut = 300 * w
n_cut_noise_spectral = 30
n_cut_mats = 100
n_cut_noise = 100
w_free = 10 * w
# Parameters 
ll = 1 * w
w0 = w
gamma = 2.1 * w0
Gamma = gamma / 2.
Omega = cmath.sqrt(w0**2 - Gamma**2)
T = 5 / w
beta = 1. / w
vec_p = [0,gamma * ll**2]
vec_w = [Omega + 1j * Gamma,-Omega + 1j * Gamma]
vec_w = regularize_poles(vec_w)
# Lists
t_list = np.linspace(-T,T,200+1) # odd to allow a middle point.
w_list = np.linspace(0.1,w_cut,100)

# Correlations
C_num = correlation_numerical(J_poly(vec_p,vec_w),t_list,beta,w_cut,n_cut_noise_spectral,n_cut_noise)
C_brownian = correlation_single_brownian(w0,gamma,ll,t_list,beta,n_cut_mats,n_cut_noise_spectral,n_cut_noise)

# Save
pickle_out = open("./PM_3/Tests/Data/check_classical.dat",'wb')
dict = {}
dict_brownian = C_brownian.save(dict)
dict_num = C_num.save(dict)
to_save = [dict_brownian,dict_num]
pickle.dump(to_save,pickle_out)
pickle_out.close()

# Plots
fig, axs = plt.subplots(2, 2,figsize=(14,7))
fig.suptitle("Check Correlations", fontsize=16)

axs[0,0].plot(t_list,np.real(C_num.C_s),color='black',label='Num. Real')
axs[0,0].plot(t_list,np.real(C_brownian.C_s),color='red',linestyle='--',label='An. Real')

axs[1,0].plot(t_list,np.imag(C_num.C_s),color='black',label='Num. Imag.')
axs[1,0].plot(t_list,np.imag(C_brownian.C_s),color='red',linestyle='--',label='An. Imag.')


axs[0,0].set_xlabel('$t$')
axs[0,0].set_ylabel('$Re[C(t)]$')
axs[0,0].set_title("Symmetric Correlations (real part)")
axs[0,0].legend(loc='lower right')
axs[1,0].set_xlabel('$t$')
axs[1,0].set_ylabel('$Im[C(t)]$')
axs[1,0].set_title("Symmetric Correlations (imaginary part)")
axs[1,0].legend()
#axs[1,0].legend(loc='lower right')

axs[0,1].plot(t_list,np.real(C_num.C_as),'black',label='Num. Real')
axs[0,1].plot(t_list,np.real(C_brownian.C_as),'r--',label='An. Real')

axs[1,1].plot(t_list,np.imag(C_num.C_as),'black',label='Num. Imag')
axs[1,1].plot(t_list,np.imag(C_brownian.C_as),'r--',label='An. Imag')

axs[0,1].set_xlabel('$t$')
axs[0,1].set_ylabel('$Re[C(t)]$')
axs[0,1].set_title("Antisymmetric Correlations (real part)")
axs[0,1].legend()
axs[1,1].set_xlabel('$t$')
axs[1,1].set_ylabel('$Im[C(t)]$')
axs[1,1].set_title("Antisymmetric Correlations (imaginary part)")
axs[1,1].legend()


plt.subplots_adjust(wspace=0.4,hspace=0.4)



plt.show()