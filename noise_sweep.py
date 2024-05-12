from direct_vqe import *
from dm_vqe import *

from utensils import *


# ansatz = Uccsd_Spin_Sym(hamiltonian.num_qubits)



noise_list = [10**(-6), 10**(-5), 10**(-4), 5*10**(-4),
               10**(-3), 2*10**(-3), 5*10**(-3), 10**(-2)]

noise_list_five = noise_list*5

hamiltonian = Hamiltonian('H4_8_R2.0.txt')
ansatz = Uccsd_Spin_Sym(hamiltonian.num_qubits)



for noise in noise_list_five:

    error = pauli_error([('X',noise/3),('Y',noise/3),
                     ('Z',noise/3),('I',1-noise)])
    num_electrons = 4
    h4_vqe_dir = Dir_VQE(hamiltonian, 4, ansatz, error)
    with open('dir_log_temp.txt','a') as f:
        f.write('\n\n\n')
        f.write('Noise strength: ')
        f.write(str(noise))
        f.write('\n\n\n')
    print('ground state energy is: ',
           hamiltonian.groundstate_energy())
    print('number of parameters: ',
           ansatz.num_params)
    print('noise: ', noise)
    params, energy, fidelity = h4_vqe_dir.run()
    fidelity_list = h4_vqe_dir.fidelity_iter_list
    energy_list = h4_vqe_dir.energy_iter_list
    with open('h4_varnoise_dir_depolarize.txt','a') as f:
        f.write("Noise strength: ")
        f.write(str(noise))
        f.write('    ')
        np.savetxt(f,np.array([[noise,energy,fidelity]]))
    with open('h4_varnoise_dir_depolarize_log.txt','a') as f:
        f.write("Noise strength: ")
        f.write(str(noise))
        f.write('    ')
        f.write("Noise: ")
        f.write(str(noise))
        f.write('\n')
        for i in range(len(fidelity_list)):
            f.write("iter:")
            f.write(str(i))
            f.write('   ')
            line = [energy_list[i], fidelity_list[i]]
            line_arr = np.array(line)
            line_arr = np.array([line_arr])
            np.savetxt(f,line_arr)
    
    error = pauli_error([('X',noise/3),('Y',noise/3),
                     ('Z',noise/3),('I',1-noise)])
    num_electrons = 4
    h4_vqe_dm = DM_VQE(hamiltonian, 4, ansatz, error)
    with open('dm_log_temp.txt','a') as f:
        f.write('\n\n\n')
        f.write('Noise strength: ')
        f.write(str(noise))
        f.write('\n\n\n')
    print('ground state energy is: ',
           hamiltonian.groundstate_energy())
    print('number of parameters: ',
           h4_vqe_dm.num_params)
    print('noise: ', noise)
    params, energy, fidelity = h4_vqe_dm.run()
    fidelity_list = h4_vqe_dm.fidelity_iter_list
    energy_list = h4_vqe_dm.energy_iter_list
    purity_list = h4_vqe_dm.purity_iter_list
    with open('h4_varnoise_dm_depolarize.txt','a') as f:
        f.write("Noise strength: ")
        f.write(str(noise))
        f.write('    ')
        np.savetxt(f,np.array([[noise,energy,fidelity]]))
    with open('h4_varnoise_dm_depolarize_log.txt','a') as f:
        f.write("Noise strength: ")
        f.write(str(noise))
        f.write('    ')
        f.write("Noise: ")
        f.write(str(noise))
        f.write('\n')
        for i in range(len(fidelity_list)):
            f.write("iter:")
            f.write(str(i))
            f.write('   ')
            line = [energy_list[i]]
            line.append(fidelity_list[i])
            line = line + purity_list[i]
            line_arr = np.array(line)
            line_arr = np.array([line_arr])
            np.savetxt(f,line_arr)
   




# for noise in noise_list_five:
#     error = pauli_error([('X',noise/3),('Y',noise/3),
#                      ('Z',noise/3),('I',1-noise)])
#     num_electrons = 4
#     h4_vqe_dm = DM_VQE(hamiltonian, 4, ansatz, error)
#     print('ground state energy is: ',
#            hamiltonian.groundstate_energy())
#     print('number of parameters: ',
#            h4_vqe_dm.num_params)
#     print('noise: ', noise)
#     params, energy, fidelity = h4_vqe_dm.run()
#     fidelity_list = h4_vqe_dm.fidelity_iter_list
#     energy_list = h4_vqe_dm.energy_iter_list
#     purity_list = h4_vqe_dm.purity_iter_list
#     with open('h4_dm_depolarize_c','ab') as f:
#         np.savetxt(f,np.array([[noise,energy,fidelity]]))
#     with open('h4_dm_depolarize_c_log','a') as f:
#         f.write("Noise: ")
#         f.write(str(noise))
#         f.write('\n')
#         for i in range(len(fidelity_list)):
#             f.write("iter:")
#             f.write(str(i))
#             f.write('   ')
#             line = [energy_list[i]]
#             line.append(fidelity_list[i])
#             line = line + purity_list[i]
#             line_arr = np.array(line)
#             line_arr = np.array([line_arr])
#             np.savetxt(f,line_arr)


