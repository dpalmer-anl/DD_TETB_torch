import torch
import torch.nn as nn
from typing import Union, List, Dict, Tuple, Callable
from NeighborList import get_neighbor_list
import numpy as np


def get_onsite_energy(atom_centered_descriptors:torch.Tensor, model):
    """get onsite energy
    returns: onsite_energy: tensor of shape [n_atoms, n_orbitals per atom]"""
    onsite_energy = model(atom_centered_descriptors)
    return onsite_energy

def get_bond_integral(pairwise_descriptors:torch.Tensor, model):
    """get bond integrals energy
    returns: bond_integrals: tensor of shape [n_pairs, n_orbitals per atom]"""
    bond_integrals = model(pairwise_descriptors)
    return bond_integrals


def create_bond_integral_from_model(model: Callable,
                                    descriptor_type: str = 'scalar') -> Callable:
    """
    Create a bond integral function from a neural network or other model.
    
    Parameters:
    -----------
    model : callable
        A PyTorch model that takes descriptors and outputs bond integrals
    descriptor_type : str
        Type of descriptor input:
        - 'scalar': descriptor is [npairs] (e.g., distance)
        - 'vector': descriptor is [npairs, d] (e.g., multi-dimensional features)
        - 'raw': descriptor passed directly to model
        
    Returns:
    --------
    bond_integral_func : callable
        Function that takes descriptors and returns bond integral values [npairs]
        
    Example:
    --------
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 1))
    >>> V_ppσ_func = create_bond_integral_from_model(model, descriptor_type='scalar')
    >>> 
    >>> # Use with distance descriptor
    >>> dist = torch.norm(disp, dim=1)
    >>> bond_integral = V_ppσ_func(dist)
    """
    
    if descriptor_type == 'scalar':
        def bond_integral_func(descriptors):
            # descriptors: [npairs]
            return model(descriptors.unsqueeze(1)).squeeze(1)
    
    elif descriptor_type == 'vector':
        def bond_integral_func(descriptors):
            # descriptors: [npairs, d]
            out = model(descriptors)
            return out.squeeze(1) if out.dim() > 1 else out
    
    elif descriptor_type == 'raw':
        def bond_integral_func(descriptors):
            # descriptors: any shape
            return model(descriptors)
    
    else:
        raise ValueError(f"Unknown descriptor_type: {descriptor_type}")
    
    return bond_integral_func


def create_simple_bond_integral(V0: float, 
                                decay: float, 
                                r0: float = 0.0,
                                form: str = 'exponential',
                                descriptor_is_distance: bool = True) -> Callable:
    """
    Create a simple parametric bond integral function.
    
    Parameters:
    -----------
    V0 : float
        Bond integral strength (eV)
    decay : float
        Decay length (Angstrom) or power exponent
    r0 : float, optional
        Reference distance (Angstrom)
    form : str
        Form of bond integral: 'exponential', 'power_law', 'morse'
    descriptor_is_distance : bool, optional
        If True, descriptor is assumed to be distance [npairs]
        If False, descriptor is assumed to be displacement [npairs, 3]
        
    Returns:
    --------
    bond_integral_func : callable
        Function that takes descriptors and returns bond integrals [npairs]
        
    Examples:
    ---------
    >>> # When descriptor is distance
    >>> V_ppσ = create_simple_bond_integral(-2.7, 0.5, r0=1.42, 
    ...                                     form='exponential', 
    ...                                     descriptor_is_distance=True)
    >>> dist = torch.norm(disp, dim=1)
    >>> bond_int = V_ppσ(dist)
    
    >>> # When descriptor is displacement
    >>> V_ppσ = create_simple_bond_integral(-2.7, 0.5, r0=1.42, 
    ...                                     form='exponential', 
    ...                                     descriptor_is_distance=False)
    >>> bond_int = V_ppσ(disp)
    """
    
    if form == 'exponential':
        def bond_integral_func(descriptors):
            if descriptor_is_distance:
                dist = descriptors  # [npairs]
            else:
                dist = torch.norm(descriptors, dim=1)  # [npairs, 3] -> [npairs]
            return V0 * torch.exp(-(dist - r0) / decay)
    
    elif form == 'power_law':
        # decay is used as the power n
        def bond_integral_func(descriptors):
            if descriptor_is_distance:
                dist = descriptors
            else:
                dist = torch.norm(descriptors, dim=1)
            return V0 * (r0 / dist) ** decay
    
    elif form == 'morse':
        # decay is used as alpha
        def bond_integral_func(descriptors):
            if descriptor_is_distance:
                dist = descriptors
            else:
                dist = torch.norm(descriptors, dim=1)
            return V0 * (torch.exp(-2*decay*(dist-r0)) - 2*torch.exp(-decay*(dist-r0)))
    
    else:
        raise ValueError(f"Unknown form: {form}")
    
    return bond_integral_func

def create_hopping_function_MLP(model: nn.Module,basis: Union[str, List[str]],
                                bond_integral_names = ['Vssσ','Vspσ', 'Vppσ', 'Vppπ','Vsdσ', 'Vpdσ', 'Vpdπ', 'Vddσ', 'Vddπ', 'Vddδ']
                                ) -> Callable: 
    """
    Create a hopping function that takes descriptors and displacements.
    assumes order of bond integrals is [Vssσ, Vspσ, Vppσ, Vppπ, Vsdσ, Vpdσ, Vpdπ, Vddσ, Vddπ, Vddδ]
    Parameters:
    -----------
    model: nn.Module
        MLPModel that takes descriptors and returns bond integrals
    basis: Union[str, List[str]]
        Orbital basis specification (e.g., 's', 'sp', 'spd', ['s', 'px', 'py', 'pz'])
    
    Returns:
    --------
    hopping_function: callable
        Function that takes descriptors and displacements and returns hopping matrix elements
    """
    orbital_list = _parse_orbital_basis(basis)
    n_orbitals = len(orbital_list)
    
    def hopping_function(descriptors: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        npairs = len(disp)
        
        # Compute bond integrals using user-defined functions
        # Functions take descriptors (user-defined format) model input shape must match number of descriptors features
        bond_int_vals = model(descriptors)
        bond_integrals = {}
        for n in range(n_orbitals):
            bond_integrals[bond_integral_names[n]] = bond_int_vals[:, n] # [npairs]
        
        # Perform Slater-Koster transformation using displacements
        # SK transform needs direction cosines from disp
        hopping_matrix = SK_transform(disp, bond_integrals, orbital_list)
        # Returns [npairs, n_orbitals, n_orbitals]
        
        # Flatten to [npairs, n_orbitals²]
        hoppings = hopping_matrix.reshape(npairs, -1)
        
        return hoppings
        
    return hopping_function

def create_hopping_function(bond_integral_functions: Dict[str, Callable], 
                            basis: Union[str, List[str]]) -> Callable:
    """
    Create a hopping function that takes descriptors and displacements.
    
    Descriptors are used to compute bond integrals (can be anything user-defined).
    Displacements are used for Slater-Koster transformations (direction cosines).
    
    Parameters:
    -----------
    bond_integral_functions : dict of callable
        Dictionary mapping bond integral names (e.g., 'V_ssσ', 'V_ppσ', 'V_ppπ') to functions.
        Each function takes descriptors and returns bond integral values [npairs].
        
        Function signature: func(descriptors: Tensor) -> Tensor[npairs]
        
        The descriptors can be:
        - Distance: Tensor[npairs]
        - Displacement: Tensor[npairs, 3]  
        - Complex descriptors: Tensor[npairs, n_features]
        - Any user-defined format
        
    basis : str or list of str
        Orbital basis specification (e.g., 's', 'sp', 'spd', ['s', 'px', 'py', 'pz'])
        
    Returns:
    --------
    hopping_function : callable
        Function with signature:
            (descriptors: Tensor, disp: Tensor[npairs, 3]) -> Tensor[npairs, norb²]
        
        - descriptors: Used to compute bond integrals (user-defined)
        - disp: Displacement vectors for Slater-Koster transform
        
    Examples:
    ---------
    >>> # Example 1: Descriptors are distances
    >>> def V_ppσ_func(descriptors):
    ...     dist = descriptors  # [npairs]
    ...     return -2.7 * torch.exp(-dist / 0.5)
    >>> 
    >>> bond_funcs = {'V_ppσ': V_ppσ_func, 'V_ppπ': V_ppπ_func}
    >>> hopping_func = create_hopping_function(bond_funcs, basis='pz')
    >>> 
    >>> # Use with distance descriptors
    >>> disp, i, j, di, dj = get_neighbor_list(positions, cell, cutoff)
    >>> dist = torch.norm(disp, dim=1)
    >>> hoppings = hopping_func(dist, disp)
    
    >>> # Example 2: Descriptors are displacements themselves
    >>> def V_ppσ_func(descriptors):
    ...     disp_desc = descriptors  # [npairs, 3]
    ...     dist = torch.norm(disp_desc, dim=1)
    ...     return -2.7 * torch.exp(-dist / 0.5)
    >>> 
    >>> hoppings = hopping_func(disp, disp)  # Same tensor used twice
    
    >>> # Example 3: Descriptors are complex features
    >>> descriptors = compute_behler_descriptors(...)  # [npairs, n_features]
    >>> hoppings = hopping_func(descriptors, disp)
    """
    
    # Parse basis
    orbital_list = _parse_orbital_basis(basis)
    n_orbitals = len(orbital_list)
    
    def hopping_function(descriptors: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        """
        Compute hopping parameters from descriptors and displacements (fully vectorized).
        
        Parameters:
        -----------
        descriptors : torch.Tensor
            Descriptors for computing bond integrals
            Shape depends on user's choice (e.g., [npairs], [npairs, 3], [npairs, n_features])
        disp : torch.Tensor [npairs, 3]
            Displacement vectors for Slater-Koster transformations
        
        Returns:
        --------
        hoppings : torch.Tensor [npairs, n_orbitals²]
            Hopping matrix elements for each pair
        """
        npairs = len(disp)
        
        # Compute bond integrals using user-defined functions
        # Functions take descriptors (user-defined format)
        bond_integrals = {}
        for name, func in bond_integral_functions.items():
            bond_integrals[name] = func(descriptors)  # [npairs]
        
        # Perform Slater-Koster transformation using displacements
        # SK transform needs direction cosines from disp
        hopping_matrix = SK_transform(disp, bond_integrals, orbital_list)
        # Returns [npairs, n_orbitals, n_orbitals]
        
        # Flatten to [npairs, n_orbitals²]
        hoppings = hopping_matrix.reshape(npairs, -1)
        
        return hoppings
    
    return hopping_function



def SK_transform(disp: torch.Tensor, bond_integrals: Dict[str, torch.Tensor], basis: List[str]) -> torch.Tensor:
    """
    Perform Slater-Koster transformation on bond integrals.
    
    Parameters:
    -----------
    disp : torch.Tensor
        Displacement vectors [npairs, 3]
    bond_integrals : dict of torch.Tensor
        Dictionary mapping bond integral names (e.g., 'V_ssσ', 'V_ppσ', 'V_ppπ') to tensors [npairs]
    basis : list of str
        List of orbital names (e.g., ['s', 'px', 'py', 'pz'])
        
    Returns:
    --------
    hoppings : torch.Tensor
        Hopping matrix [npairs, n_orbitals, n_orbitals]
    """
    npairs = len(disp)
    n_orbitals = len(basis)
    
    # Calculate direction cosines
    dist = torch.norm(disp, dim=1, keepdim=True)
    dist = torch.clamp(dist, min=1e-10)  # Avoid division by zero
    l = disp[:, 0] / dist.squeeze()
    m = disp[:, 1] / dist.squeeze()
    n = disp[:, 2] / dist.squeeze()
    
    # Initialize hopping matrix
    hoppings = torch.zeros((npairs, n_orbitals, n_orbitals), 
                          dtype=disp.dtype, device=disp.device)
    
    # Fill hopping matrix using Slater-Koster rules
    for i, orbital1 in enumerate(basis):
        for j, orbital2 in enumerate(basis):
            hopping = _calculate_slater_koster_integral(
                orbital1, orbital2, l, m, n, bond_integrals
            )
            hoppings[:, i, j] = hopping
    
    return hoppings

def get_hellman_feynman(density_matrix: torch.Tensor,
                             kpoint: torch.Tensor, disp: torch.Tensor,
                             grad_hop: torch.Tensor,
                             hop_i: torch.Tensor, hop_j: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized Hellman-Feynman force calculation following F = -ρ * dH/dR 
    
    The Hellman-Feynman theorem states: F_i = -<ψ|dH/dR_i|ψ> 
    where:
    - ρ is the density matrix: ρ_uv = Σ_i^Nocc c*_ui c_vi
    - dH/dR_i is the derivative of the Hamiltonian with respect to position R_i
    
    Args:
        density_matrix: density matrix [norbs, norbs]
        kpoint: K-point vector [3]
        disp: Displacement vectors [npairs, 3]
        grad_hop: Hopping gradients. Includes gradients of onsite and hopping terms [npairs] or [npairs*n_orbitals^2]
        hop_i: Orbital indices i [npairs*n_orbitals^2]
        hop_j: Orbital indices j [npairs*n_orbitals^2]
    
    Returns:
        Forces: Hellman-Feynman forces [natoms, 3]
    """
    nhops = len(hop_i)
    npairs = len(disp)
    n_orb_sq = nhops // npairs
    
    # Expand grad_hop if needed
    if grad_hop.numel() == npairs * 3:
        # grad_hop is [npairs, 3], need to expand to match hop_i/hop_j
        grad_hop = grad_hop.reshape(npairs, 3)
        grad_hop = grad_hop.repeat_interleave(n_orb_sq, dim=0)  # [nhops, 3]
    elif grad_hop.numel() == npairs:
        # grad_hop is [npairs], need to expand and add dimension
        grad_hop = grad_hop.unsqueeze(1).expand(-1, 3)  # [npairs, 3]
        grad_hop = grad_hop.repeat_interleave(n_orb_sq, dim=0)  # [nhops, 3]
    else:
        grad_hop = grad_hop.reshape(nhops, 3)
    
    # Expand disp to match hop_i/hop_j
    disp_expanded = disp.repeat_interleave(n_orb_sq, dim=0)  # [nhops, 3]
    
    # grad_hop contains ∂hopping/∂descriptor where descriptor = R_j - R_i
    # For chain rule: ∂E/∂R_i = ∂E/∂descriptor * ∂descriptor/∂R_i = ∂E/∂descriptor * (-1)
    #                 ∂E/∂R_j = ∂E/∂descriptor * ∂descriptor/∂R_j = ∂E/∂descriptor * (+1)
    
    # Calculate phase factors for each pair
    phases = torch.exp(1.0j * torch.sum(kpoint.unsqueeze(0) * disp_expanded, dim=1))  # [nhops]
    
    # Get density matrix elements for i->j
    rho_ij = density_matrix[hop_i, hop_j]  # [nhops]
    rho_ji = density_matrix[hop_j, hop_i]  # [nhops] (Hermitian conjugate)
    
    # Hellmann-Feynman theorem: dE = Tr[ρ * dH]
    # dH/dR = (∂t/∂descriptor) * (∂descriptor/∂R) * e^{ik·r} + t * (∂e^{ik·r}/∂R)
    # 
    # Note: Currently we only include the first term (hopping derivative).
    # The second term (phase derivative) can be added for improved accuracy:
    #   ∂e^{ik·r}/∂R = ik * e^{ik·r} * (∂r/∂R)
    # This Berry connection term is important for high-accuracy force calculations
    
    # Weight for forward hopping (i->j)
    weight_ij = (rho_ij * phases)  # [nhops] complex
    # Weight for backward hopping (j->i) - conjugate
    weight_ji = (rho_ji * phases.conj())  # [nhops] complex
    
    # Combine both directions
    weight = weight_ij + weight_ji  # [nhops] complex
    
    # Force contributions from ∂t/∂descriptor
    f_descriptor = weight.unsqueeze(1) * grad_hop.to(weight.dtype)  # [nhops, 3] complex
    
    # Determine natoms from orbital indices
    natoms = max(hop_i.max().item(), hop_j.max().item()) + 1
    if n_orb_sq > 1:
        natoms = natoms // int(n_orb_sq ** 0.5)  # Convert from n_orbitals back to n_atoms
    
    # Initialize forces tensor as COMPLEX - will be summed over k-points before taking real part
    Forces = torch.zeros((natoms, 3), device=density_matrix.device, dtype=torch.complex64)
    
    # Convert orbital indices to atom indices
    n_orb_per_atom = int(n_orb_sq ** 0.5)
    atom_i_idx = hop_i // n_orb_per_atom  # [nhops]
    atom_j_idx = hop_j // n_orb_per_atom  # [nhops]
    
    # Apply chain rule for displacement descriptors:
    # descriptor = R_j - R_i, so ∂descriptor/∂R_i = -1 and ∂descriptor/∂R_j = +1
    # 
    # Hellmann-Feynman: F_I = -∂E/∂R_I = -Tr[ρ * ∂H/∂R_I]
    # Since ∂H_ij/∂R_I = (∂t_ij/∂descriptor) * (∂descriptor/∂R_I) * e^{ik·r}:
    #   F_atom_i = -Tr[ρ_ij * (∂t/∂descriptor) * (-1) * e^{ik·r}] = +Tr[ρ_ij * (∂t/∂descriptor) * e^{ik·r}]
    #   F_atom_j = -Tr[ρ_ij * (∂t/∂descriptor) * (+1) * e^{ik·r}] = -Tr[ρ_ij * (∂t/∂descriptor) * e^{ik·r}]
    
    # Accumulate forces on atom i (gets +f_descriptor)
    Forces.index_add_(0, atom_i_idx, +f_descriptor)
    # Accumulate forces on atom j (gets -f_descriptor, satisfying Newton's 3rd law)  
    Forces.index_add_(0, atom_j_idx, -f_descriptor)
    
    return Forces

def _SK_pd_interaction(p_orbital: str, d_orbital: str, l: torch.Tensor, m: torch.Tensor, 
                       n: torch.Tensor, V_pdσ: torch.Tensor, V_pdπ: torch.Tensor) -> torch.Tensor:
    """Calculate p-d Slater-Koster interactions."""
    sqrt3 = torch.tensor(3.0).sqrt()
    
    # px-d interactions
    if p_orbital == "px":
        if d_orbital == "dxy":
            return sqrt3 * l * l * m * V_pdσ + m * (1 - 2*l*l) * V_pdπ
        elif d_orbital == "dxz":
            return sqrt3 * l * l * n * V_pdσ + n * (1 - 2*l*l) * V_pdπ
        elif d_orbital == "dyz":
            return sqrt3 * l * m * n * V_pdσ - 2 * l * m * n * V_pdπ
        elif d_orbital == "dx2-y2":
            return sqrt3/2 * l * (l*l - m*m) * V_pdσ + l * (1 - l*l + m*m) * V_pdπ
        elif d_orbital == "dz2":
            return l * (n*n - (l*l + m*m)/2) * V_pdσ - sqrt3 * l * n * n * V_pdπ
    
    # py-d interactions
    elif p_orbital == "py":
        if d_orbital == "dxy":
            return sqrt3 * m * m * l * V_pdσ + l * (1 - 2*m*m) * V_pdπ
        elif d_orbital == "dxz":
            return sqrt3 * m * l * n * V_pdσ - 2 * m * l * n * V_pdπ
        elif d_orbital == "dyz":
            return sqrt3 * m * m * n * V_pdσ + n * (1 - 2*m*m) * V_pdπ
        elif d_orbital == "dx2-y2":
            return sqrt3/2 * m * (l*l - m*m) * V_pdσ - m * (1 + l*l - m*m) * V_pdπ
        elif d_orbital == "dz2":
            return m * (n*n - (l*l + m*m)/2) * V_pdσ - sqrt3 * m * n * n * V_pdπ
    
    # pz-d interactions
    elif p_orbital == "pz":
        if d_orbital == "dxy":
            return sqrt3 * n * l * m * V_pdσ - 2 * n * l * m * V_pdπ
        elif d_orbital == "dxz":
            return sqrt3 * n * n * l * V_pdσ + l * (1 - 2*n*n) * V_pdπ
        elif d_orbital == "dyz":
            return sqrt3 * n * n * m * V_pdσ + m * (1 - 2*n*n) * V_pdπ
        elif d_orbital == "dx2-y2":
            return sqrt3/2 * n * (l*l - m*m) * V_pdσ - n * (l*l - m*m) * V_pdπ
        elif d_orbital == "dz2":
            return n * (n*n - (l*l + m*m)/2) * V_pdσ + sqrt3 * n * (l*l + m*m) * V_pdπ
    
    return torch.zeros_like(l)


def _SK_dd_interaction(d_orbital1: str, d_orbital2: str, l: torch.Tensor, m: torch.Tensor, 
                       n: torch.Tensor, V_ddσ: torch.Tensor, V_ddπ: torch.Tensor, 
                       V_ddδ: torch.Tensor) -> torch.Tensor:
    """Calculate d-d Slater-Koster interactions."""
    
    # For simplicity, implement only diagonal and most common off-diagonal terms
    # Full d-d table is quite extensive (25 unique combinations)
    
    if d_orbital1 == d_orbital2:
        if d_orbital1 == "dxy":
            return 3*l*l*m*m*V_ddσ + (l*l + m*m - 4*l*l*m*m)*V_ddπ + (n*n + l*l*m*m)*V_ddδ
        elif d_orbital1 == "dxz":
            return 3*l*l*n*n*V_ddσ + (l*l + n*n - 4*l*l*n*n)*V_ddπ + (m*m + l*l*n*n)*V_ddδ
        elif d_orbital1 == "dyz":
            return 3*m*m*n*n*V_ddσ + (m*m + n*n - 4*m*m*n*n)*V_ddπ + (l*l + m*m*n*n)*V_ddδ
        elif d_orbital1 == "dx2-y2":
            return 3/4*(l*l - m*m)**2*V_ddσ + (l*l + m*m - (l*l - m*m)**2)*V_ddπ + (n*n + (l*l - m*m)**2/4)*V_ddδ
        elif d_orbital1 == "dz2":
            return (n*n - (l*l + m*m)/2)**2*V_ddσ + 3*n*n*(l*l + m*m)*V_ddπ + 3/4*(l*l + m*m)**2*V_ddδ
    
    # Off-diagonal terms (implement key ones)
    elif (d_orbital1 == "dxy" and d_orbital2 == "dx2-y2") or (d_orbital1 == "dx2-y2" and d_orbital2 == "dxy"):
        return 3*torch.tensor(3.0).sqrt()/2 * l*m*(l*l - m*m)*V_ddσ + \
               l*m*(1 - 2*(l*l - m*m))*V_ddπ - l*m*(1 - (l*l - m*m)/2)*V_ddδ
    
    elif (d_orbital1 == "dxy" and d_orbital2 == "dxz") or (d_orbital1 == "dxz" and d_orbital2 == "dxy"):
        return 3*l*l*m*n*V_ddσ + l*n*(1 - 4*l*m)*V_ddπ + l*n*(l*m - 1)*V_ddδ
    
    elif (d_orbital1 == "dxy" and d_orbital2 == "dyz") or (d_orbital1 == "dyz" and d_orbital2 == "dxy"):
        return 3*m*m*l*n*V_ddσ + m*n*(1 - 4*l*m)*V_ddπ + m*n*(l*m - 1)*V_ddδ
    
    elif (d_orbital1 == "dxz" and d_orbital2 == "dyz") or (d_orbital1 == "dyz" and d_orbital2 == "dxz"):
        return 3*l*m*n*n*V_ddσ + l*m*(1 - 4*n*n)*V_ddπ + l*m*(n*n - 1)*V_ddδ
    
    # Add more as needed...
    return torch.zeros_like(l)


def _parse_orbital_basis(orbital_basis: Union[str, List[str]]) -> List[str]:
    """
    Parse orbital basis specification into a list of orbital names.
    
    Parameters:
    -----------
    orbital_basis : str or list of str
        Orbital basis specification
        
    Returns:
    --------
    orbital_list : list of str
        List of orbital names
    """
    
    if isinstance(orbital_basis, str):
        if orbital_basis == "s":
            return ["s"]
        elif orbital_basis == "sp":
            return ["s", "px", "py", "pz"]
        elif orbital_basis == "spd":
            return ["s", "px", "py", "pz", "dxy", "dxz", "dyz", "dx2-y2", "dz2"]
        elif orbital_basis == "p":
            return ["px", "py", "pz"]
        elif orbital_basis == "d":
            return ["dxy", "dxz", "dyz", "dx2-y2", "dz2"]
        else:
            # Check if it's a single orbital name
            valid_orbitals = ["s", "px", "py", "pz", "dxy", "dxz", "dyz", "dx2-y2", "dz2"]
            if orbital_basis in valid_orbitals:
                return [orbital_basis]
            else:
                raise ValueError(f"Unknown orbital basis: {orbital_basis}")
    elif isinstance(orbital_basis, list):
        # Validate orbital names
        valid_orbitals = ["s", "px", "py", "pz", "dxy", "dxz", "dyz", "dx2-y2", "dz2"]
        for orb in orbital_basis:
            if orb not in valid_orbitals:
                raise ValueError(f"Unknown orbital: {orb}")
        return orbital_basis
    else:
        raise TypeError("orbital_basis must be str or list of str")


def _calculate_slater_koster_integral(
    orbital1: str, 
    orbital2: str, 
    l: torch.Tensor, 
    m: torch.Tensor, 
    n: torch.Tensor, 
    bond_integrals: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Calculate Slater-Koster hopping integral between two orbitals.
    
    Parameters:
    -----------
    orbital1 : str
        First orbital name
    orbital2 : str
        Second orbital name
    l, m, n : torch.Tensor
        Direction cosines (x, y, z components of unit vector) [npairs]
    bond_integrals : dict
        Dictionary of bond integrals, each a tensor [npairs]
        
    Returns:
    --------
    hopping : torch.Tensor
        Hopping integral [npairs]
    """
    
    # Determine orbital types
    def get_orbital_type(orbital):
        if orbital == "s":
            return "s"
        elif orbital in ["px", "py", "pz"]:
            return "p"
        elif orbital in ["dxy", "dxz", "dyz", "dx2-y2", "dz2"]:
            return "d"
        else:
            raise ValueError(f"Unknown orbital: {orbital}")
    
    orb1_type = get_orbital_type(orbital1)
    orb2_type = get_orbital_type(orbital2)
    
    # Get zero tensor with correct shape for default returns
    zero = torch.zeros_like(l)
    sqrt3 = torch.tensor(3.0).sqrt()
    
    # s-s interactions
    if orb1_type == "s" and orb2_type == "s":
        return bond_integrals.get('V_ssσ', zero)
    
    # s-p interactions
    elif (orb1_type == "s" and orb2_type == "p") or (orb1_type == "p" and orb2_type == "s"):
        V_spσ = bond_integrals.get('V_spσ', zero)
        if orbital1 == "s":
            if orbital2 == "px":
                return l * V_spσ
            elif orbital2 == "py":
                return m * V_spσ
            elif orbital2 == "pz":
                return n * V_spσ
        else:  # orbital2 == "s"
            if orbital1 == "px":
                return l * V_spσ
            elif orbital1 == "py":
                return m * V_spσ
            elif orbital1 == "pz":
                return n * V_spσ
    
    # p-p interactions
    elif orb1_type == "p" and orb2_type == "p":
        V_ppσ = bond_integrals.get('V_ppσ', zero)
        V_ppπ = bond_integrals.get('V_ppπ', zero)
        
        if orbital1 == orbital2:
            # Same orbital (px-px, py-py, pz-pz)
            if orbital1 == "px":
                return l*l * V_ppσ + (1 - l*l) * V_ppπ
            elif orbital1 == "py":
                return m*m * V_ppσ + (1 - m*m) * V_ppπ
            elif orbital1 == "pz":
                return n*n * V_ppσ + (1 - n*n) * V_ppπ
        else:
            # Different p orbitals
            if (orbital1 == "px" and orbital2 == "py") or (orbital1 == "py" and orbital2 == "px"):
                return l*m * (V_ppσ - V_ppπ)
            elif (orbital1 == "px" and orbital2 == "pz") or (orbital1 == "pz" and orbital2 == "px"):
                return l*n * (V_ppσ - V_ppπ)
            elif (orbital1 == "py" and orbital2 == "pz") or (orbital1 == "pz" and orbital2 == "py"):
                return m*n * (V_ppσ - V_ppπ)
    
    # s-d interactions
    elif (orb1_type == "s" and orb2_type == "d") or (orb1_type == "d" and orb2_type == "s"):
        V_sdσ = bond_integrals.get('V_sdσ', zero)
        
        if orbital1 == "s":
            d_orbital = orbital2
        else:
            d_orbital = orbital1
            
        if d_orbital == "dxy":
            return sqrt3 * l * m * V_sdσ
        elif d_orbital == "dxz":
            return sqrt3 * l * n * V_sdσ
        elif d_orbital == "dyz":
            return sqrt3 * m * n * V_sdσ
        elif d_orbital == "dx2-y2":
            return sqrt3/2 * (l*l - m*m) * V_sdσ
        elif d_orbital == "dz2":
            return (n*n - (l*l + m*m)/2) * V_sdσ
    
    # p-d interactions
    elif (orb1_type == "p" and orb2_type == "d") or (orb1_type == "d" and orb2_type == "p"):
        V_pdσ = bond_integrals.get('V_pdσ', zero)
        V_pdπ = bond_integrals.get('V_pdπ', zero)
        
        if orb1_type == "p":
            p_orbital, d_orbital = orbital1, orbital2
        else:
            p_orbital, d_orbital = orbital2, orbital1
        
        return _SK_pd_interaction(p_orbital, d_orbital, l, m, n, V_pdσ, V_pdπ)
    
    # d-d interactions
    elif orb1_type == "d" and orb2_type == "d":
        V_ddσ = bond_integrals.get('V_ddσ', zero)
        V_ddπ = bond_integrals.get('V_ddπ', zero)
        V_ddδ = bond_integrals.get('V_ddδ', zero)
        
        return _SK_dd_interaction(orbital1, orbital2, l, m, n, V_ddσ, V_ddπ, V_ddδ)
    
    # Unknown combination
    return zero


def get_recip_cell(cell: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized reciprocal cell calculation with Intel GPU acceleration.
    Handles both 2D and 3D cells.
    """
    a1 = cell[0, :]
    a2 = cell[1, :]
    a3 = cell[2, :]
    
    volume = torch.dot(a1, torch.cross(a2, a3, dim=0))
    
    # Handle 2D case where volume is near zero
    volume = torch.where(torch.abs(volume) < 1e-10, torch.tensor(1.0, device=cell.device), volume)
    
    b1 = 2 * torch.pi * torch.cross(a2, a3, dim=0) / volume
    b2 = 2 * torch.pi * torch.cross(a3, a1, dim=0) / volume
    b3 = 2 * torch.pi * torch.cross(a1, a2, dim=0) / volume
    
    return torch.stack([b1, b2, b3])

def k_uniform_mesh(mesh_size: Tuple[int, int, int], device: torch.device = None) -> torch.Tensor:
    """
    PyTorch-optimized uniform k-mesh generation with Intel GPU acceleration.
    """
    if device is None:
        device = torch.device("cpu")
    
    nx, ny, nz = mesh_size
    
    # Create coordinate arrays
    x = torch.linspace(0, 1, nx, device=device, dtype=torch.float32)
    y = torch.linspace(0, 1, ny, device=device, dtype=torch.float32)
    z = torch.linspace(0, 1, nz, device=device, dtype=torch.float32)
    
    # Create meshgrid
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    # Reshape to final format
    k_vec = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    
    return k_vec

def k_path(sym_pts: torch.Tensor, nk: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch-optimized k-path generation with Intel GPU acceleration.
    """
    # number of nodes
    k_list=sym_pts
    n_nodes=k_list.shape[0]

    mesh_step = nk//(n_nodes-1)
    mesh = torch.linspace(0,1,mesh_step)
    step = (torch.arange(0,mesh_step,1)/mesh_step)

    kvec = torch.zeros((0,3))
    knode = torch.zeros(n_nodes)
    for i in range(n_nodes-1):
        n1 = k_list[i,:]
        n2 = k_list[i+1,:]
        diffq = torch.outer((n2 - n1),  step).T + n1

        dn = torch.linalg.norm(n2-n1)
        knode[i+1] = dn + knode[i]
        if i==0:
            kvec = torch.vstack((kvec,diffq))
        else:
            kvec = torch.vstack((kvec,diffq))
    kvec = torch.vstack((kvec,k_list[-1,:]))

    dk_ = torch.zeros(kvec.shape[0])
    for i in range(1,kvec.shape[0]):
        dk_[i] = torch.linalg.norm(kvec[i,:]-kvec[i-1,:]) + dk_[i-1]

    return (kvec,dk_, knode)


def expand_atom_to_orbital_indices(atom_i: torch.Tensor, 
                                   atom_j: torch.Tensor,
                                   hoppings: torch.Tensor,
                                   n_orbitals: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expand atom indices to orbital indices for multiple orbitals per atom.
    
    Parameters:
    -----------
    atom_i : torch.Tensor [npairs]
        Atom indices for first atom in each pair
    atom_j : torch.Tensor [npairs]
        Atom indices for second atom in each pair
    hoppings : torch.Tensor [npairs, n_orbitals * n_orbitals]
        Hopping matrix flattened for each pair
    n_orbitals : int
        Number of orbitals per atom
        
    Returns:
    --------
    hop_i : torch.Tensor [npairs * n_orbitals * n_orbitals]
        Orbital indices for first orbital in each hopping
    hop_j : torch.Tensor [npairs * n_orbitals * n_orbitals]
        Orbital indices for second orbital in each hopping
    hop_values : torch.Tensor [npairs * n_orbitals * n_orbitals]
        Hopping values (flattened)
        
    Example:
    --------
    For 2 atoms with 3 orbitals each (p orbitals):
    - Atom 0 has orbitals [0, 1, 2]
    - Atom 1 has orbitals [3, 4, 5]
    
    If atom_i = [0] and atom_j = [1], this expands to:
    - hop_i = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    - hop_j = [3, 4, 5, 3, 4, 5, 3, 4, 5]
    """
    npairs = len(atom_i)
    
    if n_orbitals == 1:
        # Simple case: one orbital per atom
        return atom_i, atom_j, hoppings.reshape(-1)
    
    # For each pair, create all orbital-to-orbital connections
    # Shape: [npairs, n_orbitals, n_orbitals]
    hop_matrix = hoppings.reshape(npairs, n_orbitals, n_orbitals)
    
    # Create orbital indices for each atom
    # atom_i orbital indices: atom_i * n_orbitals + [0, 1, ..., n_orbitals-1]
    # Shape: [npairs, n_orbitals]
    orb_i = (atom_i.unsqueeze(1) * n_orbitals + 
             torch.arange(n_orbitals, device=atom_i.device).unsqueeze(0))
    
    # Shape: [npairs, n_orbitals]
    orb_j = (atom_j.unsqueeze(1) * n_orbitals + 
             torch.arange(n_orbitals, device=atom_j.device).unsqueeze(0))
    
    # Expand to all pairs of orbitals
    # Shape: [npairs, n_orbitals, n_orbitals]
    hop_i_expanded = orb_i.unsqueeze(2).expand(npairs, n_orbitals, n_orbitals)
    hop_j_expanded = orb_j.unsqueeze(1).expand(npairs, n_orbitals, n_orbitals)
    
    # Flatten everything
    hop_i = hop_i_expanded.reshape(-1)
    hop_j = hop_j_expanded.reshape(-1)
    hop_values = hop_matrix.reshape(-1)
    
    return hop_i, hop_j, hop_values
