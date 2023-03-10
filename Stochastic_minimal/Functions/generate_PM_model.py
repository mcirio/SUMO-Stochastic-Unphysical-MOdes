import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'Functions' )
sys.path.append( mymodule_dir )

import numpy as np
from quantum_functions import Hamiltonian_single
from quantum_functions import create_PM_single
from quantum_functions import generate_initial_state
from quantum_functions import tensor_identities
from quantum_functions import generate_L
from quantum_functions import generate_operators

def generate_PM_model(H_S,psi0_S,s,ordered_PM_parameters,obs_list,space_system_dim,space_modes_dim):
    N_modes = len(ordered_PM_parameters)
    H_S = tensor_identities(H_S,N_modes,space_modes_dim)
    s = tensor_identities(s,N_modes,space_modes_dim)
    H_xi = Hamiltonian_single(s)
    operators_list = generate_operators(N_modes,space_system_dim,space_modes_dim)
    L = generate_L(H_S,s,ordered_PM_parameters,operators_list)
    beta_PM = 'inf' # The PM are at T=0 by construction! This is not the beta of the environment
    psi0 = generate_initial_state(psi0_S,beta_PM,N_modes,space_modes_dim)
    new_obs_list = []
    for obs in obs_list: 
        new_obs_list.append(tensor_identities(obs,N_modes,space_modes_dim))
    c_list = []

    return H_S, s, H_xi, L, psi0, new_obs_list, c_list