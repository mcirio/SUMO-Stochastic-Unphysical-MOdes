import numpy as np
import pickle
import matplotlib.pyplot as plt

pickle_in = open("./PM_3/Tests/Data/check_classical.dat",'rb')
[dict_an,dict_num] = pickle.load(pickle_in)
pickle_in.close()

C_an_C_as = dict_an['C_as'] 
C_an_C_as_2 = dict_an['C_as_2'] 
C_an_C_s = dict_an['C_s'] 
C_an_C_s_2 = dict_an['C_s_2'] 
C_an_coeff_list = dict_an['coeff_list'] 
C_an_C_s_reconstructed = dict_an['C_s_reconstructed'] 
C_an_C_s_stochastic = dict_an['C_s_stochastic'] 
C_an_beta = dict_an['beta'] 
C_an_n_cut_mats = dict_an['n_cut_mats'] 
C_an_n_cut_noise_spectral = dict_an['n_cut_noise_spectral'] 
C_an_n_cut_noise= dict_an['n_cut_noise'] 
C_an_w_free = dict_an['w_free'] 
t_list = dict_an['t_list']




C_num_C_as = dict_num['C_as'] 
C_num_C_s = dict_num['C_s'] 
C_num_C = dict_num['C'] 
C_num_coeff_list = dict_num['coeff_list']
C_num_C_s_reconstructed = dict_num['C_s_reconstructed'] 
C_num_C_s_stochastic = dict_num['C_s_stochastic'] 
C_num_error_symm = dict_num['error_symm']
C_num_beta = dict_num['beta'] 
C_num_w_cut = dict_num['w_cut'] 
C_num_n_cut_noise_spectral = dict_num['n_cut_noise_spectral'] 
C_num_n_cut_noise= dict_num['n_cut_noise'] 


# Plots
fig, axs = plt.subplots(2, 2,figsize=(14,7))
fig.suptitle("Check Correlations", fontsize=16)

axs[0,0].plot(t_list,np.real(C_num_C_s),color='black',label='Num. Real')
axs[0,0].plot(t_list,np.real(C_num_C_s_reconstructed),'b',linestyle='-',label='Rec. Real, n_cut={}'.format(C_num_n_cut_noise_spectral))
axs[0,0].plot(t_list,np.real(C_an_C_s_reconstructed),'b',linestyle='-',label='Rec.(an) Real, n_cut={}'.format(C_an_n_cut_noise_spectral))
axs[0,0].plot(t_list,np.real(C_num_C_s_stochastic),color='g',linestyle='-',label='Stoch. Real, n_cut={}, n_noise={}'.format(C_num_n_cut_noise_spectral,C_num_n_cut_noise))
axs[0,0].fill_between(t_list, np.real(C_num_C_s_stochastic)-C_num_error_symm, np.real(C_num_C_s_stochastic)+C_num_error_symm,alpha=0.2, edgecolor='darkgreen', facecolor='lightgreen')
axs[0,0].plot(t_list,np.real(C_an_C_s),color='red',linestyle='--',label='An. Real')
axs[0,0].plot(t_list,np.real(C_an_C_s_2),color='darkgrey',linestyle='dotted',label='An. Real 2')
#symm),color='darkgrey',linestyle='dotted',label='An. Real 2')

axs[1,0].plot(t_list,np.imag(C_num_C_s),color='black',label='Num. Imag.')
axs[1,0].plot(t_list,np.imag(C_num_C_s_reconstructed),'b',linestyle='-',label='Rec. Imag., n_cut={}'.format(C_num_n_cut_noise_spectral))
axs[1,0].plot(t_list,np.imag(C_an_C_s_reconstructed),'b',linestyle='-',label='Rec.(an) Imag., n_cut={}'.format(C_an_n_cut_noise_spectral))
axs[1,0].plot(t_list,np.imag(C_num_C_s_stochastic),color='g',linestyle='-',label='Stoch. Imag., n_cut={}, n_noise={}'.format(C_num_n_cut_noise_spectral,C_num_n_cut_noise))
axs[1,0].fill_between(t_list, np.imag(C_num_C_s_stochastic)-C_num_error_symm, np.imag(C_num_C_s_stochastic)+C_num_error_symm,alpha=0.2, edgecolor='darkgreen', facecolor='lightgreen')
axs[1,0].plot(t_list,np.imag(C_an_C_s),color='red',linestyle='--',label='An. Imag.')
axs[1,0].plot(t_list,np.imag(C_an_C_s_2),color='darkgrey',linestyle='dotted',label='An. Real 2')
#axs[0,0].plot(t_list,np.imag(C_an_test_antisymm),color='darkgrey',linestyle='dotted',label='An. Real 2')


axs[0,0].set_xlabel('$t$')
axs[0,0].set_ylabel('$Re[C(t)]$')
axs[0,0].set_title("Symmetric Correlations (real part)")
axs[0,0].legend(loc='lower right')
axs[1,0].set_xlabel('$t$')
axs[1,0].set_ylabel('$Im[C(t)]$')
axs[1,0].set_title("Symmetric Correlations (imaginary part)")
axs[1,0].legend()
#axs[1,0].legend(loc='lower right')

axs[0,1].plot(t_list,np.real(C_num_C_as),'black',label='Num. Real')
axs[0,1].plot(t_list,np.real(C_an_C_as),'r--',label='An. Real')
axs[0,1].plot(t_list,np.real(C_an_C_as_2),'darkgrey',linestyle='dotted',label='An. Real 2')

axs[1,1].plot(t_list,np.imag(C_num_C_as),'black',label='Num. Imag')
axs[1,1].plot(t_list,np.imag(C_an_C_as),'r--',label='An. Imag')
axs[1,1].plot(t_list,np.imag(C_an_C_as_2),'darkgrey',linestyle='dotted',label='An. Imag 2')

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