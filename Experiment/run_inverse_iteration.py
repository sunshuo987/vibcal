import numpy as np
import scipy
import cProfile
import pstats
import os
from typing import List
from pytreenet.operators import TensorProduct, Hamiltonian
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.util import SVDParameters
from pytreenet.util.misc_functions import orthogonalise_cholesky, orthogonalise_gram_schmidt, orthogonalise_gep, linear_combination
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.dmrg.als import AlternatingLeastSquares
# TODO:from pytreenet.contractions.state_state_contraction import add, block_diag_list, linear_combination_list

from potentials import get_potential_energy_CH3CN, get_potential_energy_CH3CN_harmonic
from utils import get_name_list, orbitals_array2str, get_orbitals_indices_first, get_energy_clusters, get_ttno
from utils_ch3cn import random_mps, random_ttns, random_t3ns, random_binary_mctdh, random_threetree, random_twotree, random_threetree2, random_mps_harmonic_oscillator_0, random_threetree_harmonic_oscillator_0
import matplotlib.pyplot as plt
import pandas as pd
import time
from copy import deepcopy
np.random.seed=42

def run_block_inverse_iteration(ham_pad: Hamiltonian, ttnss: List[TreeTensorNetworkState], save_to_path: str,num_sweeps = 5, svd_params: SVDParameters = SVDParameters(max_bond_dim=10, rel_tol=1e-5, total_tol=1e-5)):
      
    es_ii = []
    ttno = TreeTensorNetworkOperator.from_hamiltonian(ham_pad, ttnss[0], dtype=np.float64)
    es_i = [ttns.operator_expectation_value(ttno) for ttns in ttnss]
    print("Energy", es_i)
    es_ii.append(es_i)
    shift_avg = np.mean(es_i).real
    for i in range(num_sweeps):
        #===========
        
        # shift = np.mean([ttns.operator_expectation_value(ttno) for ttns in ttnss])
        shift_avg = np.mean(es_i).real
        
        ham_pad_shift = deepcopy(ham_pad)
        ham_pad_shift.conversion_dictionary[f'{shift_avg}I9'] = -np.eye(9)*shift_avg
        shift_term = TensorProduct({'site0':f'{shift_avg}I9','site1':'I7','site2':'I9','site3':'I9','site4':'I9','site5':'I9','site6':'I7','site7':'I7','site8':'I9','site9':'I9','site10':'I27','site11':'I27'})
        ham_pad_shift.add_term(shift_term)
        shifted_ttno = TreeTensorNetworkOperator.from_hamiltonian(ham_pad_shift, ttnss[0], dtype=np.float64)
        #===========
        print("Energy shift", shift_avg)
        es_i = []
        for j in range(len(ttnss)):
            # if np.isclose(es_ii[-1][j],shift_avg,atol=1e-4):
            #     shift = shift_avg+1e-4
            #     print("Shift", shift)
            # else:
            #     shift = shift_avg
            solver = AlternatingLeastSquares(shifted_ttno, state_x = ttnss[j], state_b = ttnss[j], num_sweeps = 1, max_iter = 600, svd_params = svd_params, site = "one-site", residual_rank = 2)
            solver.run()
            ene = solver.state_x.operator_expectation_value(ttno).real
            ttnss[j] = solver.state_x
            es_i.append(ene )
            print(f"Sweep {i}, State {j}, Energy {ene}, dtype {ttnss[j].tensors['site0'].dtype}, max bond dim, {max(ttnss[j].bond_dims().values())}")
        es_ii.append(es_i)
        if len(ttnss) > 1:
            ttnss = orthogonalise_gep(ttno, ttnss, min(svd_params.max_bond_dim, max(ttnss[0].bond_dims().values())), num_sweeps = 1)

            for k in range(len(ttnss)):
                print(f"Sweep {i}, State {k}, Energy {ttnss[k].operator_expectation_value(ttno)}")
        else:
            ttnss[0] = linear_combination(ttnss, [1.0], min(svd_params.max_bond_dim, max(ttnss[0].bond_dims().values())), num_sweeps=1)
    return np.array(es_ii).T.tolist(), ttnss

def main(results_dir, num_sweeps = 10, max_bond_dim = 12):
    
    profiler = cProfile.Profile()
    profiler.enable()
    # results_dir = "./Results/ch3cn/threetree/max_bond_dim_12"
    ene_lobpcg = np.loadtxt(f"{results_dir}/lobpcg_energies.csv", delimiter=",")[:, -1]
    clusters = get_energy_clusters(ene_lobpcg,0.005,10) 
    print("Clusters", clusters)
    state_dir = f"{results_dir}/states"
    
    reference_energy = pd.read_csv("./Experiment/ref_energies.csv")
    
    N = [9,7,9,9,9,9,7,7,9,9,27,27]
    # name_list, conversion_dict = get_name_list(N, get_potential_energy_CH3CN)
    # new_name_list = [{k: v for k, v in d.items() if k != 'coeff'} for d in name_list]
    # terms = [TensorProduct(new_name_list[i]) for i in range(len(new_name_list))]
    # conversion_dict['I1'] = np.eye(1)
    
    state = TreeTensorNetworkState.load(f"{state_dir}/lobpcg_state_0")
    # ham = Hamiltonian(terms, conversion_dictionary=conversion_dict)
    # ham_pad = ham.pad_with_identities(state, symbolic=True)
    # ttno = TreeTensorNetworkOperator.from_hamiltonian(ham_pad, state, dtype=np.float64)
    ttno, ham_pad = get_ttno(N, state, get_potential_energy_CH3CN, True)
    print("expected energy", state.operator_expectation_value(ttno))
    cl_state_cnt = 75
    energies = []
    for ix, cl in enumerate(clusters[27:28]):
        # cl+=clusters[22]
        time_start = time.time()
        print(f"Cluster {ix}", cl)
        ttnss = [TreeTensorNetworkState.load(state_dir + f"/lobpcg_state_{i}") for i in cl]
        
        es_ii, ttnss = run_block_inverse_iteration(ham_pad, ttnss, state_dir, num_sweeps = num_sweeps, svd_params = SVDParameters(max_bond_dim=max_bond_dim, rel_tol=1e-10, total_tol=1e-10))
        energies.append(es_ii)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
        for k in range(len(es_ii)):
            plt.plot(abs((es_ii[k])-reference_energy["Fine_Energy"][cl_state_cnt + k])*1000, label="lobpcg")
            plt.yscale("log")
            plt.plot(range(len(es_ii[k])), np.ones(len(es_ii[k])), label="reference")
            plt.ylabel("Energy difference (cm-1)")
            plt.xlabel("Iteration")
            plt.legend()
            plt.title(f"inverse-iteration_energies_{cl_state_cnt + k}")
            plt.savefig(f"{results_dir}/inverse-iteration_energies_{cl_state_cnt + k}_wresi_20.png")
            plt.close()
            # plt.show()
            
            ttnss[k].save(f"{state_dir}/inverse_iteration_state_{cl_state_cnt + k}_wresi_20")
            print("bond dim", k, ttnss[k].bond_dims().values())    
        cl_state_cnt += len(cl)
    energies = np.array(energies)
    np.savetxt(f"{results_dir}/inverse_iteration_energies_cache_20.csv", np.concatenate(energies,axis = 0), delimiter=",")

    energies = np.concatenate(energies,axis = 0)
    np.savetxt(f"{results_dir}/inverse_iteration_energies_20.csv", np.array(energies), delimiter=",")
    profiler.disable()
    profiler.dump_stats(f"{results_dir}/profile_output_ii_20.prof")
    # print(f"Profiling complete. Stats saved to threetree/max_bond_dim_{max_bond_dim}/profile_output_ii.prof")

if __name__ == "__main__":
#     main(results_dir = "./Results/250722/mps/max_bond_dim_12", 
#          num_sweeps = 5, max_bond_dim = 20)
    main(results_dir = "./Results/ch3cn/mps/max_bond_dim_12", 
         num_sweeps = 6, max_bond_dim = 20)
    # main(results_dir = "./Results/ch3cn/threetree/max_bond_dim_12", 
        #  num_sweeps = 6, max_bond_dim = 20)