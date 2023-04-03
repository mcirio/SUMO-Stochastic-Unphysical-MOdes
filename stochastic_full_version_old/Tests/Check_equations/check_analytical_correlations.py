import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..\..', 'Classes' )
sys.path.append( mymodule_dir )
mymodule_dir = os.path.join( script_dir, '..\..', 'Functions' )
sys.path.append( mymodule_dir )

from regularize_poles import regularize_poles
from J_poly import J_poly
from correlation_analytical import correlation_analytical
from correlation_numerical import correlation_numerical

import numpy as np
import matplotlib.pyplot as plt
import pickle


# Units
w = 2 * np.pi
# Cuts 
w_cut = 100 * w
n_cut_noise_spectral = 30
n_cut_mats = 100
n_cut_noise = 100
w_free = 10 * w
# Parameters 
T = 5 / w
beta = 4. / w
Omega = .4 * w
Gamma = .9 * w
gamma = 2 * Gamma
ll = .3 * w**1.5
Omega_S = w
#vec_p = [0,1]
#vec_w = [(.4 + .9*1j)*w,1j*w]
vec_p = [0,2 * Gamma * ll**2]
vec_w = [Omega + 1j*Gamma,0.6*1j*Gamma]
vec_w = regularize_poles(vec_w)
J = J_poly(vec_p,vec_w)



# Lists
t_list = np.linspace(-T,T,200+1) # odd to allow a middle point.

# Correlations
C_an = correlation_analytical(vec_p,vec_w,t_list,beta,n_cut_mats,n_cut_noise_spectral,n_cut_noise,w_free)
C_num = correlation_numerical(J_poly(vec_p,vec_w),t_list,beta,w_cut,n_cut_noise_spectral,n_cut_noise)
w_list = C_an.w_list
J_list = [J(w) for w in w_list]

# Save
pickle_out = open("./PM_3/Tests/Data/check_classical.dat",'wb')
dict = {}
dict_an = C_an.save(dict)
dict_num = C_num.save(dict)
to_save = [dict_an,dict_num]
pickle.dump(to_save,pickle_out)
pickle_out.close()

# Plots
fig, axs = plt.subplots(2, 2,figsize=(14,7))
fig.suptitle("Check Correlations", fontsize=16)

axs[0,0].plot(t_list,np.real(C_num.C_s),color='black',label='Num. Real')
axs[0,0].plot(t_list,np.real(C_num.C_s_reconstructed),'b',linestyle='-',label='Rec. Real, n_cut={}'.format(n_cut_noise_spectral))
axs[0,0].plot(t_list,np.real(C_an.C_s_reconstructed),'b',linestyle='-',label='Rec.(an) Real, n_cut={}'.format(n_cut_noise_spectral))
axs[0,0].plot(t_list,np.real(C_num.C_s_stochastic),color='g',linestyle='-',label='Stoch. Real, n_cut={}, n_noise={}'.format(n_cut_noise_spectral,n_cut_noise))
axs[0,0].fill_between(t_list, np.real(C_num.C_s_stochastic)-C_num.error_symm, np.real(C_num.C_s_stochastic)+C_num.error_symm,alpha=0.2, edgecolor='darkgreen', facecolor='lightgreen')
axs[0,0].plot(t_list,np.real(C_an.C_s),color='red',linestyle='--',label='An. Real')
axs[0,0].plot(t_list,np.real(C_an.C_s_2),color='darkgrey',linestyle='dotted',label='An. Real 2')
#symm),color='darkgrey',linestyle='dotted',label='An. Real 2')

axs[1,0].plot(t_list,np.imag(C_num.C_s),color='black',label='Num. Imag.')
axs[1,0].plot(t_list,np.imag(C_num.C_s_reconstructed),'b',linestyle='-',label='Rec. Imag., n_cut={}'.format(n_cut_noise_spectral))
axs[1,0].plot(t_list,np.imag(C_an.C_s_reconstructed),'b',linestyle='-',label='Rec.(an) Imag., n_cut={}'.format(n_cut_noise_spectral))
axs[1,0].plot(t_list,np.imag(C_num.C_s_stochastic),color='g',linestyle='-',label='Stoch. Imag., n_cut={}, n_noise={}'.format(n_cut_noise_spectral,n_cut_noise))
axs[1,0].fill_between(t_list, np.imag(C_num.C_s_stochastic)-C_num.error_symm, np.imag(C_num.C_s_stochastic)+C_num.error_symm,alpha=0.2, edgecolor='darkgreen', facecolor='lightgreen')
axs[1,0].plot(t_list,np.imag(C_an.C_s),color='red',linestyle='--',label='An. Imag.')
axs[1,0].plot(t_list,np.imag(C_an.C_s_2),color='darkgrey',linestyle='dotted',label='An. Real 2')
#axs[0,0].plot(t_list,np.imag(C_an.test_antisymm),color='darkgrey',linestyle='dotted',label='An. Real 2')


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
axs[0,1].plot(t_list,np.real(C_an.C_as),'r--',label='An. Real')
axs[0,1].plot(t_list,np.real(C_an.C_as_2),'darkgrey',linestyle='dotted',label='An. Real 2')

axs[1,1].plot(t_list,np.imag(C_num.C_as),'black',label='Num. Imag')
axs[1,1].plot(t_list,np.imag(C_an.C_as),'r--',label='An. Imag')
axs[1,1].plot(t_list,np.imag(C_an.C_as_2),'darkgrey',linestyle='dotted',label='An. Imag 2')

axs[0,1].set_xlabel('$t$')
axs[0,1].set_ylabel('$Re[C(t)]$')
axs[0,1].set_title("Antisymmetric Correlations (real part)")
axs[0,1].legend()
axs[1,1].set_xlabel('$t$')
axs[1,1].set_ylabel('$Im[C(t)]$')
axs[1,1].set_title("Antisymmetric Correlations (imaginary part)")
axs[1,1].legend()

fig2, axs2 = plt.subplots(1, 2,figsize=(14,7))
fig.suptitle("Check Correlations", fontsize=16)

axs2[0].plot(np.real(C_an.C_Q))
axs2[0].plot(np.imag(C_an.C_Q))
axs2[0].plot(np.real(C_an.C_Q2),'r--')
axs2[0].plot(np.imag(C_an.C_Q2),'r--')
axs2[1].plot(np.real(C_an.C_s_plus_fs))
axs2[1].plot(np.imag(C_an.C_s_plus_fs))
axs2[1].plot(np.real(C_an.C_s_plus_fs2),'r--')
axs2[1].plot(np.imag(C_an.C_s_plus_fs2),'r--')

axs2[1].plot(np.real(C_an.C_rec),'g--')
axs2[1].plot(np.imag(C_an.C_rec),'g--')

fig3, axs3 = plt.subplots(1, 1,figsize=(14,7))
fig.suptitle("Fouerier", fontsize=16)

axs3.plot(w_list,np.real(np.abs(C_an.c_list)))
#axs2[2].plot(np.real(np.imag(C_an.c_list)))

left, bottom, width, height = [0.588, 0.575, 0.4, 0.4]
axs4 = fig3.add_axes([left, bottom, width, height])

#fig3, axs3 = plt.subplots(1, 1,figsize=(16,7))
#axs3.set_title('Spectral density', fontsize=30)
axs4.plot(w_list,J_list,'b',linewidth='6')
axs4.axvline(Omega_S,color='k',linestyle='--')
axs4.set_xlabel('frequency [$\omega_S$]', fontsize=28)
axs4.set_ylabel('$J(\omega) [(\gamma\lambda^2)^{1/4}]$', fontsize=28)
axs4.tick_params(axis='both', which='major', labelsize=20)
axs4.tick_params(axis='both', which='minor', labelsize=15)
plt.tight_layout(h_pad=0, w_pad=5)

plt.subplots_adjust(wspace=0.4,hspace=0.4)

plt.show()