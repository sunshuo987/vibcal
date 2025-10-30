from typing import List
import numpy as np
from pytreenet.core.node import Node
from pytreenet.random.random_node import random_tensor_node
from pytreenet.ttns import TTNS
from pytreenet.special_ttn import MatrixProductTree
from pytreenet.util.tensor_splitting import SplitMode
from Experiment.utils import get_harmonic_oscillator_orbitals, single_voxel_block_diag
import warnings
from scipy import optimize

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

def create_node_mapping(node_order: list) -> dict:
    """
    Create a node mapping dictionary where keys are 0-2 and values are from the provided list.
    
    Args:
        node_order (list): List of integers specifying the new order of nodes
        
    Returns:
        dict: Mapping dictionary where keys are 0-2 and values are from node_order
    """
    if len(node_order) != 3:
        raise ValueError("node_order must contain exactly 3 values")
        
    if sorted(node_order) != list(range(3)):
        raise ValueError("node_order must contain all integers from 0 to 2")
        
    return {i: node_order[i] for i in range(3)}


def random_mps(physical_dim: List[int], virtual_dim: int, weights: List[float], node_order: List[int]) -> TTNS:
    """
    Generate a random MPS with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_mps: TTNS
            The random MPS.
    """
    nsite = len(physical_dim)
    shapes = [[virtual_dim]]+[[virtual_dim,virtual_dim]]*(nsite-2)+[[virtual_dim]]
    if nsite != len(shapes):
        raise ValueError("The length of physical_dim and shapes must be the same")
    sort_idx = np.argsort(node_order)
    shapes = [shapes[i] for i in sort_idx]
    new_shapes = []
    for i in range(nsite):
        new_shapes.append(shapes[i]+[physical_dim[i]])
        
    nodes = [random_tensor_node(shape, identifier="site"+str(i))
             for i, shape in enumerate(new_shapes)]
    
    node_mapping = create_node_mapping(node_order)
    random_mps = TTNS()
    random_mps.add_root(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1])
    random_mps.add_child_to_parent(nodes[node_mapping[1]][0],nodes[node_mapping[1]][1],0,f"site{node_mapping[0]}",0)
    for i in range(nsite-2):
        random_mps.add_child_to_parent(nodes[node_mapping[i+2]][0],nodes[node_mapping[i+2]][1],0,f"site{node_mapping[i+1]}",1)
    random_mps.canonical_form(f"site{node_mapping[nsite-1]}", mode=SplitMode.KEEP)
    # random_mps.init_multistate_center()
    return random_mps

def random_mps_harmonic_oscillator(physical_dim: List[int], omega: List[float], orb_state: np.ndarray, weights: List[float], node_order: List[int]) -> TTNS:
    """
    Generate a random MPS with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        omega: List[float]
            The frequency of the harmonic oscillator.
        orb_state: np.ndarray
            The state of the harmonic oscillator.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_mps: MultiTTNS
            The random MPS.
    """
    nsite = len(physical_dim)
    nneighbour = [1]+[2]*(nsite-2)+[1]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)
            random_col = np.random.randn(ho_tensors[i].shape[0],24)*1e-2
            ho_tensors[i] = np.concatenate([ho_tensors[i], random_col], axis=1)
    # for i in range(nsite):
    #     ho_tensors[i] = ho_tensors[i].reshape(new_shapes[i])
    #     ho_tensors[i] += np.random.randn(*ho_tensors[i].shape)*1e-2
    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])
        
        ho_tensors[i] = ho_tensors[i].astype(np.complex128)  # Convert to complex first
        ho_tensors[i] += (np.random.randn(*ho_tensors[i].shape) + 1j * np.random.randn(*ho_tensors[i].shape)) * 1e-2
    
    sort_idx = np.argsort(node_order)
    ho_tensors = [ho_tensors[i] for i in sort_idx]
        
    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    node_mapping = create_node_mapping(node_order)
    random_mps = TTNS()
    random_mps.add_root(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1])
    random_mps.add_child_to_parent(nodes[node_mapping[1]][0],nodes[node_mapping[1]][1],0,f"site{node_mapping[0]}",0)
    for i in range(nsite-2):
        random_mps.add_child_to_parent(nodes[node_mapping[i+2]][0],nodes[node_mapping[i+2]][1],0,f"site{node_mapping[i+1]}",1)
    random_mps.canonical_form(f"site{node_mapping[nsite-1]}", mode=SplitMode.KEEP)
    # random_mps.init_multistate_center()
    return random_mps

def random_mps_harmonic_oscillator_0(physical_dim: List[int], omega: List[float], orb_state: np.ndarray, node_order: List[int], dtype: np.dtype = np.float64) -> TTNS:
    """
    Generate a random MPS with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        omega: List[float]
            The frequency of the harmonic oscillator.
        orb_state: np.ndarray
            The state of the harmonic oscillator.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_mps: TTNS
            The random MPS.
    """
    nsite = len(physical_dim)
    nneighbour = [1]+[2]*(nsite-2)+[1]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)
    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])      
        ho_tensors[i] = ho_tensors[i].astype(dtype)  

    sort_idx = np.argsort(node_order)
    ho_tensors = [ho_tensors[i] for i in sort_idx]
    random_mps = MatrixProductTree()
    random_mps.from_tensor_list(ho_tensors, root_site=int(nsite/2))    
    # nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    # node_mapping = create_node_mapping(node_order)
    # random_mps = TTNS()
    # random_mps.add_root(nodes[node_mapping[0]][0], nodes[node_mapping[0]][1])
    # random_mps.add_child_to_parent(nodes[node_mapping[1]][0],nodes[node_mapping[1]][1],0,f"site{node_mapping[0]}",0)
    # for i in range(nsite-2):
    #     random_mps.add_child_to_parent(nodes[node_mapping[i+2]][0],nodes[node_mapping[i+2]][1],0,f"site{node_mapping[i+1]}",1)
    # random_mps.canonical_form(f"site{node_mapping[nsite-1]}", mode=SplitMode.KEEP)

    return random_mps

def random_threetree(physical_dim: List[int], virtual_dim: int, dtype: np.dtype = np.float64) -> TTNS:
    """
    Generate a random T3NS with given physical dimensions and virtual dimensions.
    """
    nsite = len(physical_dim)
    ss = nsite//3
    nneighbour = [1]+[2]*(2*ss-2)+[1]
    nneighbour.extend([2]*(nsite - 2*ss-1)+[1])

    m = virtual_dim
    shapes = [[m]] + [[m,m] for _ in range(2*ss-2)] + [[m]]
    shapes.extend([[m,m] for _ in range(nsite - 2*ss-1)])
    shapes.append([m])
    shapes_ancillary = [m,m,m,1]
    if len(physical_dim) != len(shapes):
        raise ValueError("The length of physical_dim and shapes must be the same")

    for s, d in zip(shapes, physical_dim):
        s.append(d)

    shapes.append(shapes_ancillary)
    
    nodes = [random_tensor_node(shape, identifier="site"+str(i))
             for i, shape in enumerate(shapes)]
    random_ttns = TTNS()
    random_ttns.add_root(nodes[nsite][0], nodes[nsite][1])
    cnt = 0
    
    random_ttns.add_child_to_parent(nodes[ss-1][0],nodes[ss-1][1],0,"site"+str(nsite),0)
    random_ttns.add_child_to_parent(nodes[ss][0],nodes[ss][1],0,"site"+str(nsite),1)
    random_ttns.add_child_to_parent(nodes[ss*2][0],nodes[ss*2][1],0,"site"+str(nsite),2)
    
    for i in reversed(range(ss-1)):
        random_ttns.add_child_to_parent(nodes[i][0],nodes[i][1],0,"site"+str(i+1),1)
        cnt += 1   
    for i in range(ss+1,2*ss):
        random_ttns.add_child_to_parent(nodes[i][0],nodes[i][1],0,"site"+str(i-1),1)
    for i in range(2*ss+1,nsite):
        random_ttns.add_child_to_parent(nodes[i][0],nodes[i][1],0,"site"+str(i-1),1)
    
    random_ttns.canonical_form("site"+str(nsite), mode=SplitMode.KEEP)
    # random_ttns.init_multistate_center()
    random_ttns.normalize()
    return random_ttns
    

def random_threetree_harmonic_oscillator_0(physical_dim: List[int], omega: float, orb_state: List[float], node_order: List[int], dtype: np.dtype = np.float64) -> TTNS:
    """
    Generate a random MPS with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: MultiTTNS
            The random TTNS.
    """
    nsite = len(physical_dim)
    ss = nsite//3
    nneighbour = [1]+[2]*(2*ss-2)+[1]
    nneighbour.extend([2]*(nsite - 2*ss-1)+[1])
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    m = orb_state.shape[0]
    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)

    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])     
        ho_tensors[i] = ho_tensors[i].astype(dtype) 

    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    shapes_ancillary = [[m,m,m,1]]
    nodes.extend([random_tensor_node(shape, identifier="site"+str(i+nsite), dtype=dtype)
             for i, shape in enumerate(shapes_ancillary)])
    
    random_ttns = TTNS()
    random_ttns.add_root(nodes[nsite][0], nodes[nsite][1])
    cnt = 0
    
    random_ttns.add_child_to_parent(nodes[ss-1][0],nodes[ss-1][1],0,"site"+str(nsite),0)
    random_ttns.add_child_to_parent(nodes[ss][0],nodes[ss][1],0,"site"+str(nsite),1)
    random_ttns.add_child_to_parent(nodes[ss*2][0],nodes[ss*2][1],0,"site"+str(nsite),2)
    
    for i in reversed(range(ss-1)):
        random_ttns.add_child_to_parent(nodes[i][0],nodes[i][1],0,"site"+str(i+1),1)
        cnt += 1   
    for i in range(ss+1,2*ss):
        random_ttns.add_child_to_parent(nodes[i][0],nodes[i][1],0,"site"+str(i-1),1)
    for i in range(2*ss+1,nsite):
        random_ttns.add_child_to_parent(nodes[i][0],nodes[i][1],0,"site"+str(i-1),1)
    
    random_ttns.canonical_form("site"+str(nsite), mode=SplitMode.KEEP)
    # random_ttns.init_multistate_center()
    random_ttns.normalize()
    return random_ttns

def random_threetree_harmonic_oscillator_1(physical_dim: List[int], omega: float, orb_state: List[float], node_order: List[int], dtype: np.dtype = np.float64) -> TTNS:
    """
    Generate a random MPS with given physical dimensions and virtual dimensions.
    
    Input:
        physical_dim: List[int]
            The physical dimensions of the MPS.
        virtual_dim: int
            The virtual dimensions of the MPS.
        weights: List[float]
            The weights of the MPS.
        node_order: List[int]
            The order of the nodes.
    Output:
        random_ttns: MultiTTNS
            The random TTNS.
    """
    nsite = len(physical_dim)
    ss = nsite//3
    nneighbour = [2]*(nsite-3)+[1,1,1]
    ho_tensors = get_harmonic_oscillator_orbitals(physical_dim, omega, orb_state)
    m = orb_state.shape[0]
    if orb_state.shape[0] == 1:
        for i in range(nsite):
            ho_tensors[i] = ho_tensors[i].reshape(-1,1)

    for i in range(nsite):
        ho_tensors[i] = single_voxel_block_diag(ho_tensors[i], nneighbour[i])     
        ho_tensors[i] = ho_tensors[i].astype(dtype) 

    nodes = [(Node(tensor=ho_tensor, identifier="site"+str(i)), ho_tensor) for i, ho_tensor in enumerate(ho_tensors)]
    
    shapes_ancillary = [[m,m,m,1]]
    nodes.extend([random_tensor_node(shape, identifier="site"+str(i+nsite), dtype=dtype)
             for i, shape in enumerate(shapes_ancillary)])
    
    random_ttns = TTNS()
    random_ttns.add_root(nodes[nsite][0], nodes[nsite][1])
    cnt = 0
    
    random_ttns.add_child_to_parent(nodes[0][0],nodes[0][1],0,"site"+str(nsite),0)
    random_ttns.add_child_to_parent(nodes[1][0],nodes[1][1],0,"site"+str(nsite),1)
    random_ttns.add_child_to_parent(nodes[2][0],nodes[2][1],0,"site"+str(nsite),2)
     
    for i in range(3, nsite):
        random_ttns.add_child_to_parent(nodes[i][0],nodes[i][1],0,"site"+str(i-3),1)
    # for i in range(nsite-2,nsite+1):
    #     random_ttns.add_child_to_parent(nodes[i][0],nodes[i][1],0,"site"+str(i-3),1)
    
    random_ttns.canonical_form("site"+str(0), mode=SplitMode.KEEP)
    # random_ttns.init_multistate_center()
    random_ttns.normalize()
    return random_ttns


def __main__():
    physical_dim = [3,5,7,9,3,5,7,9,3,5,7,9]
    omega = [1.0,1.0,5.0,6.0,1.0,1.0,5.0,6.0,1.0,1.0,5.0,6.0]
    orb_state = [[0,0,0,0,0,0,0,0,0,1,0,2]]
    weights = [1.0]*2
    node_order = [0,1,2,3,4,5,6,7,8,9,10,11]
    random_t3ns_harmonic_oscillator_0(physical_dim, omega, orb_state, node_order)

if __name__ == "__main__":
    __main__()