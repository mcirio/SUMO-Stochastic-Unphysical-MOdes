import numpy as np
from qutip import *
from integral import integral

###############
#Attempt to construct purely rate-based version
#follow PRA 95, 052126 equation 42 onwards
# V = c_corr_list * s 
def TimeDepRates(H_S, V, corr, tlist):
    print('Computing rates({length})'.format(length="tlist"))

    #given H_S, corr and tlist construct a superoperator for the time-dependent-rate MS.
    if not isinstance(H_S, (Qobj, QobjEvo)):
            raise TypeError("The Hamiltonian (H) must be a Qobj or QobjEvo")
            
    if not isinstance(V, (Qobj, QobjEvo)):
            raise TypeError("The coupling operator must be a Qobj or QobjEvo")                
    
    if H_S.type != "oper": 
            raise TypeError("The Hamiltonian (H) must be an operator not a state or superoperator")
    
    if V.type != "oper": 
            raise TypeError("The coupling operator must be an operator not a state or superoperator")
    print(H_S)
    energies,states = H_S.eigenstates()
    
   
    def com(V):
        return spre(V) - spost(V)
    
    C_array = np.zeros([len(tlist),len(states),len(states)], dtype = np.complex64)
    
    
    def soft(t,ws):
    
        return(corr(t)*np.exp(1.0j * ws * t))
    
    #simple version:
    
    for i in range(len(energies)):
        for j in range(len(energies)):
            for k, t in enumerate(tlist):
                Deltaij = energies[i] - energies[j]
                if V.matrix_element(states[i],states[j])!=0:
                    C_array[k, i, j] = integral(soft, Deltaij, x_i=0, x_f=t, limit=500)
          
    Vt = com(sum([QobjEvo([(V.matrix_element(states[i], states[j])*states[i] * states[j].dag()), 
                           Cubic_Spline(tlist[0],tlist[-1], C_array[:,i,j])]) 
                 for i in range(len(states)) for j in range(len(states))]))
    
    ##attempt to vectorize a bit, might be useful to fix for full PM model in the future
    #statesd = np.array([state.dag() for state in states])
    #Deltaij = np.subtract.outer(energies, energies)
    #V_array = np.array([[V.matrix_element(i, j) for j in states] for i in states])
    #nonzero = np.nonzero(V_array)
    #for k, t in enumerate(tlist):
    #    C_array[k][nonzero] = integral(soft, Deltaij[nonzero], x_i=0, x_f=t, limit=500)
                

    #QobjEvo_list = []
    #for i in range(len(states)):
    #    for j in range(len(states)):
    #        V_ij = V_array[i, j]
    #        C_ij = C_array[:, i, j]
    #        if V_ij != 0:
    #            QobjEvo_list.append(QobjEvo([(V_ij * states[i] * states[j].dag()), Cubic_Spline(tlist[0], tlist[-1], C_ij)]))
            
    #Vt = com(sum(QobjEvo_list))
    
    #EOM is \dot{rho} = -i [H_s,rho] - [V,[Vt,rho(t)]]  #approximation is rho(s) -> rho(t)
    # Where Vt = int_0^t c(s)V(s) =  \sum_ij <i|V|j> |i><j|  * (int_0^t ds c(s) e^i{e_i-e_j}s)
    L = liouvillian(QobjEvo(H_S)) - com(QobjEvo(V))*Vt
        
    return L