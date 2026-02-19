import numpy as np
import scipy
from copy import deepcopy
from typing import Callable, List, Union, Tuple
from .als import AlternatingLeastSquares
from .variational_fitting import VariationalFitting
from ..ttns.ttns_ttno.src import src_linear_combination, src_ttns_ttno_application, src_addition

from ..util.misc_functions import linear_combination, orthogonalise_gram_schmidt, orthogonalise_to
from ..util.tensor_splitting import SVDParameters

from ..ttns.ttns_ttno.zipup import zipup 
from ..contractions.state_operator_contraction import get_matrix_element
from ..ttno.ttno_class import TTNO
from ..ttns import TreeTensorNetworkState
from ..operators.hamiltonian import Hamiltonian

from ..ttns.ttns_ttno.application import apply_and_truncate, ApplicationMethod
from ..core.truncation.truncating import TruncationMethod
from ..core.addition.addition import AdditionMethod, add_two_ttns
from ..core.truncation.truncating import svd_truncation

def precond_lobpcg(shifted_ttno: TTNO, state: TreeTensorNetworkState, svd_params: SVDParameters, num_sweeps: int=2, max_iter: int=10) -> TreeTensorNetworkState:
    als = AlternatingLeastSquares(shifted_ttno, state_x = deepcopy(state), state_b = deepcopy(state), num_sweeps=num_sweeps, max_iter=max_iter, svd_params=svd_params, site='one-site', residual_rank=0)
    als.run()    
    return als.state_x

def lobpcg_single(ttno:TTNO, state_x: TreeTensorNetworkState,precond_func: Callable, svd_params: SVDParameters, max_iter: int, file_path: List[str]) -> Tuple[TreeTensorNetworkState, list[float]]:
    """
    Perform LOBPCG on a single TTNS.
    """
    num_sweeps = 2
    rayleigh = state_x.operator_expectation_value(ttno).real
    energy = [rayleigh]
    state_p = None 
    # shift = 1
    # identity_ttno = TTNO.from_hamiltonian(Hamiltonian.identity_like(state_x, shift), state_x)
    
    for i in range(max_iter):
        state_new = zipup(ttno, state_x, svd_params)
        varfit = VariationalFitting([ttno], [deepcopy(state_x)], state_new.conjugate(), num_sweeps, 100, svd_params, "one-site", [1.])
        varfit.run()
        state_r = varfit.y.conjugate()
        state_r = add_two_ttns(state_x, state_r, AdditionMethod.SRC, desired_dimension=svd_params.max_bond_dim)
        state_r.canonical_form(state_r.root_id)
        # num_r = state_r.completely_contract_tree(True)[0].flatten()
        state_r = precond_func(state_r, svd_params)
        # print("state_r",state_r.bond_dims().values())
        if len(file_path) > 0:
            state_r = orthogonalise_to(state_r, file_path, svd_params.max_bond_dim, num_sweeps)
            state_x = orthogonalise_to(state_x, file_path, svd_params.max_bond_dim, num_sweeps)
        xrp_list = [state_x, state_r]
        # print("state_x",state_x.bond_dims().values())
        if state_p is not None:
            # if len(file_path) > 0:
                # state_p = orthogonalise_to(state_p, file_path, svd_params.max_bond_dim, num_sweeps)
            xrp_list.append(state_p)
            # print("state_p",state_p.bond_dims().values())
        H = np.zeros((len(xrp_list),len(xrp_list)), dtype=np.complex128)
        M = np.zeros((len(xrp_list),len(xrp_list)), dtype=np.complex128)
        
        for j in range(len(xrp_list)):
            H[j,j] = xrp_list[j].operator_expectation_value(ttno).real
            M[j,j] = xrp_list[j].scalar_product(xrp_list[j]).real
            for k in np.arange(j+1,len(xrp_list)):
                H[j,k] = get_matrix_element(xrp_list[j].conjugate(), ttno, xrp_list[k])
                H[k,j] = H[j,k].conj()
                M[j,k] = xrp_list[k].scalar_product(xrp_list[j])
                M[k,j] = M[j,k].conj()
        ew, ev = scipy.linalg.eigh(H, M)
        ev = ev.real
        state_x = linear_combination(xrp_list, ev[:,0].tolist(), int(svd_params.max_bond_dim), num_sweeps = num_sweeps)
        # print("state_x",state_x.bond_dims().values())
        if state_p is None:
            state_p = state_r.scale(ev[1,0], inplace=False)
        else:
            state_p = linear_combination([state_r, state_p], [ev[1,0], ev[2,0]], svd_params.max_bond_dim, num_sweeps = num_sweeps)
        # xrp_list.append(state_p)
        norm = state_x.normalise()
        rayleigh = ew[0].real 
        energy.append(rayleigh)
   
    return state_x, energy

def lobpcg_block(ttno:TTNO, state_x_list: List[TreeTensorNetworkState],precond_func: Callable, svd_params: SVDParameters, max_iter: int, file_path: List[str], apply_method: ApplicationMethod | str = ApplicationMethod.SRC, trunc_method: TruncationMethod | str = TruncationMethod.SVD) -> Tuple[TreeTensorNetworkState, list[float]]:
    
    if isinstance(apply_method, str):
        apply_method = ApplicationMethod(apply_method.lower())
    if isinstance(trunc_method, str):
        trunc_method = TruncationMethod(trunc_method.lower())
    num_sweeps = 2
    n_states = len(state_x_list)
    energies = []
    rayleigh = [state.operator_expectation_value(ttno).real for state in state_x_list]
    state_p_list = None 
    
    for i in range(max_iter):
        state_r_list = []
        for ix, state_x in enumerate(state_x_list):
            state_r = apply_and_truncate(state_x, ttno, apply_method, trunc_method, app_args= (svd_params.max_bond_dim * 4,),trunc_args=(svd_params,))
            state_r.normalise()
            state_r = add_two_ttns(state_x.scale(-rayleigh[ix], inplace=False), state_r, AdditionMethod.SRC, desired_dimension=svd_params.max_bond_dim*2)
            state_r = svd_truncation(state_r, SVDParameters(svd_params.max_bond_dim, 1e-10, 1e-10))
            state_r.normalise()
            state_r.canonical_form(state_r.root_id)
            state_r = precond_func(state_r, svd_params)
            if len(file_path) > 0:
                state_r = orthogonalise_to(state_r, file_path, svd_params.max_bond_dim, num_sweeps)
                state_x = orthogonalise_to(state_x, file_path, svd_params.max_bond_dim, num_sweeps)
            state_x_list[ix] = state_x
            state_r_list.append(state_r)
            
        xrp_list = state_x_list + state_r_list
        if state_p_list is not None:
            # if len(file_path) > 0:
                # state_p = orthogonalise_to(state_p, file_path, svd_params.max_bond_dim, num_sweeps)
            xrp_list += state_p_list
        H = np.zeros((len(xrp_list),len(xrp_list)), dtype=np.float64)
        M = np.zeros((len(xrp_list),len(xrp_list)), dtype=np.float64)

        for j in range(len(xrp_list)):
            H[j,j] = xrp_list[j].operator_expectation_value(ttno).real
            M[j,j] = xrp_list[j].scalar_product(xrp_list[j]).real
            for k in np.arange(j+1,len(xrp_list)):
                H[j,k] = get_matrix_element(xrp_list[j].conjugate(), ttno, xrp_list[k]).real
                H[k,j] = H[j,k]
                M[j,k] = xrp_list[k].scalar_product(xrp_list[j]).real
                M[k,j] = M[j,k]
        ew, ev = scipy.linalg.eigh(H, M)
        ev = ev.real
        state_x_list_new = []
        state_p_list_new = []
        for n in range(n_states):
            state_x = linear_combination(xrp_list, ev[:3*n_states,n], int(svd_params.max_bond_dim), num_sweeps = num_sweeps)
            state_x_list_new.append(state_x)
            if state_p_list is None:
                state_p = linear_combination(state_r_list, ev[n_states:,n], int(svd_params.max_bond_dim), num_sweeps = num_sweeps)
            else:
                state_p = linear_combination(xrp_list[n_states:], ev[n_states:3*n_states,n], svd_params.max_bond_dim, num_sweeps = num_sweeps)
            state_p_list_new.append(state_p)
        state_x_list = state_x_list_new
        state_p_list = state_p_list_new
        rayleigh = ew[:n_states].real 
        energies.append(rayleigh)
    return state_x_list, np.array(energies).T.tolist()
   
