    
NUM_CORES = 10

import numpy as np
from sumo.Functions.progressbar import progressbar
from qutip import *

def average_dynamics(L,H_xi,xi_list,c_list,t_list,psi0,n_noise,obs_list):
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
        dynamics_average += dynamics_list[-1]
    dynamics_average = dynamics_average / n_noise
    sigma = 0 * np.array(dynamics_average)
    for dynamics in dynamics_list:
        sigma += np.array(dynamics)**2
    sigma = np.sqrt(sigma / len(dynamics_list) - np.array(dynamics_average)**2) / np.sqrt(n_noise)
    return dynamics_average, sigma, dynamics_list
    
    


def avr_me(k, L,H_xi,xi_list,c_list,t_list,psi0,n_noise,obs_list):
        xi = xi_list[k]
        L_xi = [H_xi,xi] 
        args = {}
        options = Options(num_cpus=4, atol=1e-15,nsteps = 100000000)
        #dynamics_list.append(mesolve([L,L_xi], psi0, t_list, c_list, obs_list,args=args,options=options).expect[0])
        result = mesolve([L,L_xi], psi0, t_list, c_list, obs_list,args=args,options=options).expect[0]
        return result

def average_dynamics_parallel(L,H_xi,xi_list,c_list,t_list,psi0,n_noise,obs_list):
    sigma = 1
    dynamics_average = 0
    dynamics_list = []    
    k = 0
    print("Averaging the dynamics")
    
    
        #dynamics_average += dynamics_list[-1]
	
    dynamics_list = parallel_map(avr_me, range(n_noise), task_args=(L,H_xi,xi_list,c_list,t_list,psi0,n_noise,obs_list))
    #dynamics_average = sum([res[-1] for res in dynamics_list])
    dynamics_average = sum(dynamics_list)
    
    
    dynamics_average = dynamics_average / n_noise
    sigma = 0 * np.array(dynamics_average)
    for dynamics in dynamics_list:
        sigma += np.array(dynamics)**2
    #sigma = cmath.sqrt(sigma / len(dynamics_list) - np.array(dynamics_average)**2) / np.sqrt(n_noise)
    sigma = np.sqrt(sigma / len(dynamics_list) - np.array(dynamics_average)**2) / np.sqrt(n_noise)
    #sigma = np.sqrt(sigma / len(dynamics_list) - np.array(dynamics_average)**2) / np.sqrt(n_noise)
    return dynamics_average, sigma, dynamics_list