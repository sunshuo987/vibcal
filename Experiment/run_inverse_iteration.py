import matplotlib.pyplot as plt
import pandas as pd
import time
from copy import deepcopy
import numpy as np
import cProfile

from typing import List
from pytreenet.operators import TensorProduct, Hamiltonian
from pytreenet.ttno import TreeTensorNetworkOperator
from pytreenet.util import SVDParameters
from pytreenet.util.misc_functions import orthogonalise_gep, linear_combination
from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.dmrg.als import AlternatingLeastSquares

from potentials import get_potential_energy_CH3CN
from utils import get_energy_clusters, get_ttno

np.random.seed=42

def run_block_inverse_iteration(ham_pad: Hamiltonian, ttnss: List[TreeTensorNetworkState], save_to_path: str,num_sweeps = 5, svd_params: SVDParameters = SVDParameters(max_bond_dim=10, rel_tol=1e-5, total_tol=1e-5)):
      
    es_ii = []
    ttno = TreeTensorNetworkOperator.from_hamiltonian(ham_pad, ttnss[0], dtype=np.float64)
    es_i = [ttns.operator_expectation_value(ttno) for ttns in ttnss]
    print("Energy", es_i)
    es_ii.append(es_i)

    for i in range(num_sweeps):
        shift_avg = np.mean(es_i).real
        
        ham_pad_shift = deepcopy(ham_pad)
        ham_pad_shift.conversion_dictionary[f'{shift_avg}I9'] = -np.eye(9)*shift_avg
        shift_term = TensorProduct({'site0':f'{shift_avg}I9','site1':'I7','site2':'I9','site3':'I9','site4':'I9','site5':'I9','site6':'I7','site7':'I7','site8':'I9','site9':'I9','site10':'I27','site11':'I27'})
        ham_pad_shift.add_term(shift_term)
        shifted_ttno = TreeTensorNetworkOperator.from_hamiltonian(ham_pad_shift, ttnss[0], dtype=np.float64)

        es_i = []
        for j in range(len(ttnss)):
            solver = AlternatingLeastSquares(shifted_ttno, state_x = ttnss[j], state_b = ttnss[j], num_sweeps = 1, max_iter = 1000, svd_params = svd_params, site = "one-site", residual_rank = 2)
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
    ene_lobpcg = np.loadtxt(f"{results_dir}/lobpcg_energies.csv", delimiter=",")[:, -1]
    clusters = get_energy_clusters(ene_lobpcg,0.005,10) 
    print("Clusters", clusters)
    state_dir = f"{results_dir}/states"
    
    reference_energy = pd.read_csv("./Experiment/ch3cn_ref.csv")["Energy_Ref"].values
    reference_energy[1:] = reference_energy[1:] + reference_energy[0]
    
    N = [9,7,9,9,9,9,7,7,9,9,27,27]
    state = TreeTensorNetworkState.load(f"{state_dir}/lobpcg_state_0")
    _, ham_pad = get_ttno(N, state, get_potential_energy_CH3CN, True)

    cl_state_cnt = 0
    energies = []
    for ix, cl in enumerate(clusters[0:]):
        # cl+=clusters[0]
        time_start = time.time()
        print(f"Cluster {ix}", cl)
        ttnss = [TreeTensorNetworkState.load(state_dir + f"/lobpcg_state_{i}") for i in cl]
        
        es_ii, ttnss = run_block_inverse_iteration(ham_pad, ttnss, state_dir, num_sweeps = num_sweeps, svd_params = SVDParameters(max_bond_dim=max_bond_dim, rel_tol=1e-10, total_tol=1e-10))
        energies.append(es_ii)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
        for k in range(len(es_ii)):
            plt.plot(abs(np.array(es_ii[k])*1000-reference_energy[cl_state_cnt + k]), label="inverse iteration")
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
        np.savetxt(f"{results_dir}/inverse_iteration_energies_cache_20.csv", np.concatenate(energies,axis = 0), delimiter=",")

    energies = np.concatenate(energies,axis = 0)
    np.savetxt(f"{results_dir}/inverse_iteration_energies_20.csv", np.array(energies), delimiter=",")
    profiler.disable()
    profiler.dump_stats(f"{results_dir}/profile_output_ii_20.prof")

if __name__ == "__main__":
    main(results_dir = "./Results_final/ch3cn/mps/max_bond_dim_12", 
         num_sweeps = 5, max_bond_dim = 20)
    main(results_dir = "./Results_final/ch3cn/threetree/max_bond_dim_12", 
         num_sweeps = 5, max_bond_dim = 20)
    main(results_dir = "./Results_final/ch3cn/leafonly/max_bond_dim_12", 
         num_sweeps = 5, max_bond_dim = 20)
