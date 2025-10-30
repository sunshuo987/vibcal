import numpy as np
import scipy
import cProfile
import pstats
import os

from pytreenet.operators import TensorProduct, Hamiltonian
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.util import SVDParameters
from pytreenet.util.misc_functions import linear_combination
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.dmrg.lobpcg import lobpcg_block, precond_lobpcg
from pytreenet.contractions.state_operator_contraction import get_matrix_element
from typing import List
# TODO:from pytreenet.contractions.state_state_contraction import add, block_diag_list, linear_combination_list

from potentials import get_potential_energy_harmonic
from utils import get_name_list_harmonic, orbitals_array2str, get_orbitals_indices_first, get_energy_clusters
from utils_hdo import random_mps, random_mps_harmonic_oscillator_0
from utils_harmonic import random_threetree,random_threetree_harmonic_oscillator_0,random_threetree_harmonic_oscillator_1
import matplotlib.pyplot as plt
import pandas as pd
import time

np.random.seed=42

def get_basis(ttno: TreeTensorNetworkOperator, ttnss: List[TreeTensorNetworkState], max_bond_dim: int) -> List[TreeTensorNetworkState]:
    """
    Get the basis of the TreeTensorNetworkOperator.
    """
    mat = np.zeros((len(ttnss), len(ttnss)), dtype = np.float64)
    for i in range(len(ttnss)):
        mat[i,i] = get_matrix_element(ttnss[i], ttno, ttnss[i]).real
        for j in range(i):
            mat[i,j] = get_matrix_element(ttnss[i], ttno, ttnss[j]).real
            mat[j,i] = mat[i,j]
    e,v = np.linalg.eigh(mat)        
    print("eigenvalues", e-np.diag(mat),)
    basis = []
    for i in range(len(ttnss)):    
        basis.append(linear_combination(ttnss, v[i,:], max_bond_dim, num_sweeps=2))
    for i in range(len(basis)):
        print(basis[i].operator_expectation_value(ttno) - e[i])
    return basis

def run_calculation(state_type='mps', max_bond_dim=10):
    start_time = time.time()
    N = [8,8]*32
    # node_order = [0,1,2,3,6,7,4,5,10,11,8,9]
    node_order = np.arange(len(N)).tolist()

    # Create results directory for this state type
    results_dir = f"./Results/test_harmonic/{state_type}/harmonic_oscillator/max_bond_dim_{max_bond_dim}_opt"
    states_dir = f"{results_dir}/states"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    name_list, conversion_dict = get_name_list_harmonic(len(N),N[0])
    
    new_name_list = [{k: v for k, v in d.items() if k != 'coeff'} for d in name_list]
    terms = [TensorProduct(new_name_list[i]) for i in range(len(new_name_list))]
    conversion_dict['I1'] = np.eye(1)
    shift = 121
    conversion_dict[f'{shift}I{N[0]}'] = np.eye(N[0])*shift
    ham = Hamiltonian(terms, conversion_dictionary=conversion_dict)
    
    # # Choose state type
    # if state_type == 'mps':
    #     state = random_mps(N, 8, [1.0], node_order)
    # else:  # threetree
    #     state = random_threetree(N, 2)

    w_ref,Aij = get_potential_energy_harmonic(D=len(N))
    omega = np.diag(Aij)
    
    orb_index, orb_state, orb_Es = get_orbitals_indices_first(omega,num_orb=30)
    _,_,orb_Es_ref = get_orbitals_indices_first(w_ref,num_orb=30)
    clusters = get_energy_clusters(orb_Es, 0.1, 10)
    print("Orbital energies", orb_Es)
    print("Orbital energies reference", orb_Es_ref)
    
    print("Clusters", clusters)
    states = []
    
    
    for i in range(len(orb_Es)):
        if state_type == 'mps':
            states.append(random_mps_harmonic_oscillator_0(N, omega, orb_state[i].reshape(1,-1),node_order))
            states[-1].canonical_form(states[-1].root_id)
        else:  # threetree
            states.append(random_threetree_harmonic_oscillator_0(N, omega, orb_state[i].reshape(1,-1),node_order))
            states[-1].canonical_form(states[-1].root_id)
    ham_pad = ham.pad_with_identities(states[0], symbolic=True)
    ttno = TreeTensorNetworkOperator.from_hamiltonian(ham_pad, states[0], dtype=np.float64)
    print(f"ttno dims for {state_type}:", ttno.bond_dims().values())
    # mat, _ = ttno.as_matrix()
    # print("analytical", np.linalg.eigh(mat)[0][:6])
    shift_term = TensorProduct({'site0':f'{shift}I{N[0]}','site1':f'I{N[1]}','site2':f'I{N[2]}'})

    ham_pad.add_term(shift_term)
    ttno_shift = TreeTensorNetworkOperator.from_hamiltonian(ham_pad, states[0], dtype=np.float64)
    precond_func = lambda state, svd_params: precond_lobpcg(ttno_shift, state, svd_params)
    reference_energy = pd.read_csv("./Experiment/ref_energies.csv")
    
    for i in range(len(states)):
        print("hamonic energy",states[i].operator_expectation_value(ttno))
    # states = get_basis(ttno, states[:30], max_bond_dim)
    # file_path = [f"{states_dir}/lobpcg_state_{i}" for i in range(6)]
    # energies = pd.read_csv(f"{results_dir}/lobpcg_energies.csv")
    file_path = []
    energies = []
    
    state_counter = 0 # 69
    for cl in clusters:
        # print("cluster", i, "out of", len(clusters))
        time_start = time.time()
        states_list = [states[c] for c in cl]
        states_opt_list, energies_i = lobpcg_block(ttno, states_list, precond_func, SVDParameters(max_bond_dim=max_bond_dim, renorm=True, rel_tol=1e-5, total_tol=1e-5), 8, file_path)
        time_end = time.time()
        print(f"Time taken for state {cl} lobpcg_block: {time_end - time_start} seconds")

        energies+=energies_i
        for ix, s in enumerate(states_opt_list):
            s.save(f"{states_dir}/lobpcg_state_{state_counter + ix}") 
            file_path.append(f"{states_dir}/lobpcg_state_{state_counter + ix}")
        
            plt.plot(abs(energies_i[ix]-orb_Es_ref[state_counter + ix]), label="lobpcg")
            plt.yscale("log")
            plt.plot(range(len(energies_i[ix])), np.ones(len(energies_i[ix])), label="reference")
            plt.ylabel("Energy difference (cm-1)")
            plt.xlabel("Iteration")
            plt.legend()
            plt.title(f"lobpcg_energies_{state_type}_{state_counter + ix}")
            plt.savefig(f"{results_dir}/lobpcg_energies_{state_counter + ix}.png")
            plt.close()
        print("energies", energies_i[-1], orb_Es_ref[state_counter + ix])
        state_counter += len(cl)
    
    # energies_save = np.concatenate(energies, axis=0)
        csv_file = f"{results_dir}/lobpcg_energies.csv"
        np.savetxt(csv_file, energies, delimiter=",")

    print(f"Time taken for {state_type} with max bond dim {max_bond_dim}: ", time.time() - start_time)
    return time.time() - start_time

def main():
    # Run for both MPS and threetree with separate profiling
    for max_bond_dim in [5]:
        for state_type in ['mps','threetree']:
            print(f"\nRunning calculation for {state_type} with max bond dim {max_bond_dim}...")
            profiler = cProfile.Profile()
            profiler.enable()
            run_calculation(state_type, max_bond_dim=max_bond_dim)
            profiler.disable()
            profiler.dump_stats(f"./Results/test_harmonic/{state_type}/harmonic_oscillator/max_bond_dim_{max_bond_dim}_opt/profile_output.prof")
            # print(f"Profiling complete for {state_type} with max bond dim {max_bond_dim}. Stats saved to {state_type}/water/max_bond_dim_{max_bond_dim}/profile_output.prof")

if __name__ == "__main__":
    main()
