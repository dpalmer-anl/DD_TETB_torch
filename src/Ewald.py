import torch
import ase
from NeighborList import get_neighbor_list

def get_Ewald_energy(positions: torch.Tensor, cell: torch.Tensor, charge: torch.Tensor,
                        alpha=5, nmax=3, mmax=3,rcut=6,epsilon_0=1) -> torch.Tensor:
    """get Ewald energy. Perform an ewald sum over all charges centered on atom positions.
    Perform real space sum, reciprocal space sum, and self interaction sum.
    Args:
        positions: torch.Tensor - atomic positions
        cell: torch.Tensor - cell vectors
        charge: torch.Tensor - charges on each atom
        alpha: float - Ewald screening parameter
        nmax: int - maximum real space lattice index
        mmax: int - maximum reciprocal lattice index
        rcut: float - cutoff radius
        epsilon_0: float - permittivity constant
    """

    Volume = torch.abs(torch.det(cell))
    displacement_vector, i, j, di, dj = get_neighbor_list(positions,cell,rcut)
    r = torch.norm(displacement_vector,dim=1)
    Vr, dVr_dr = potential_realsum(r, charge, i, j, alpha, epsilon_0)
    Vk, Forces_recip = potential_recipsum(positions, charge, cell, Volume, alpha, mmax, epsilon_0)
    Vs, dVs_dr = potential_selfsum(r, charge, alpha)
    V_background = -(torch.pi * torch.sum(charge**2)) / (alpha**2 * Volume * (4*torch.pi * epsilon_0))

    forces_ij_real = (dVr_dr + dVs_dr).unsqueeze(1) * displacement_vector / r.unsqueeze(1)
    Forces = torch.zeros((len(atoms),3), dtype=charge.dtype, device=charge.device)
    Forces.index_add_(0, i, forces_ij_real)
    Forces = Forces + Forces_recip
    return torch.sum(Vr + Vk + Vs + V_background), Forces

def potential_realsum(r, charge, i, j, alpha, epsilon_0):
    """real space sum"""
    E_real = torch.sum(charge[i] * charge[j] * torch.erfc(alpha * r) / r) * 0.5 * (1 / (4*torch.pi * epsilon_0))
    dE_real_dr = -charge[i] * charge[j] * torch.erfc(alpha * r) / r**2 
    return E_real, dE_real_dr

def potential_recipsum(positions, charge, cell, volume, alpha, mmax, epsilon_0):
    """
    Vectorized reciprocal space sum for Ewald summation.
    
    Args:
        positions: torch.Tensor of shape (n_atoms, 3) - atomic positions
        charge: torch.Tensor of shape (n_atoms,) - charges on each atom
        cell: torch.Tensor of shape (3, 3) - cell vectors as rows
        volume: float - cell volume
        alpha: float - Ewald screening parameter
        mmax: int - maximum reciprocal lattice index
        epsilon_0: float - permittivity constant
    
    Returns:
        E_recip: torch.Tensor - reciprocal space energy
        forces: torch.Tensor of shape (n_atoms, 3) - forces on each atom
    """
    # Compute reciprocal lattice vectors (2π * inverse transpose of cell)
    recip_cell = 2 * torch.pi * torch.linalg.inv(cell).T
    b1, b2, b3 = recip_cell[0], recip_cell[1], recip_cell[2]
    
    # Generate all reciprocal lattice vectors G within mmax range
    # Build G vectors: G = n1*b1 + n2*b2 + n3*b3
    n_range = torch.arange(-mmax, mmax + 1, device=positions.device, dtype=torch.long)
    n1, n2, n3 = torch.meshgrid(n_range, n_range, n_range, indexing='ij')
    
    # Convert to float for subsequent operations
    n1 = n1.to(positions.dtype)
    n2 = n2.to(positions.dtype)
    n3 = n3.to(positions.dtype)
    
    # Flatten and remove (0,0,0)
    n1_flat = n1.flatten()
    n2_flat = n2.flatten()
    n3_flat = n3.flatten()
    mask = ~((n1_flat == 0) & (n2_flat == 0) & (n3_flat == 0))
    
    n1_flat = n1_flat[mask]
    n2_flat = n2_flat[mask]
    n3_flat = n3_flat[mask]
    
    # Compute G vectors: shape (n_G, 3)
    G_vectors = (n1_flat.unsqueeze(1) * b1.unsqueeze(0) + 
                 n2_flat.unsqueeze(1) * b2.unsqueeze(0) + 
                 n3_flat.unsqueeze(1) * b3.unsqueeze(0))
    
    # Compute G^2 for each G vector: shape (n_G,)
    G2 = torch.sum(G_vectors**2, dim=1)
    
    # Compute G·r for all atoms and G vectors: shape (n_G, n_atoms)
    # G_vectors: (n_G, 3), positions: (n_atoms, 3)
    # G·r = G_vectors @ positions.T
    G_dot_r = torch.matmul(G_vectors, positions.T)
    
    # Compute structure factors using vectorized operations
    # S(G) = sum_i q_i * exp(i G·r_i)
    # S_real = sum_i q_i * cos(G·r_i)
    # S_imag = sum_i q_i * sin(G·r_i)
    cos_Gr = torch.cos(G_dot_r)  # shape: (n_G, n_atoms)
    sin_Gr = torch.sin(G_dot_r)  # shape: (n_G, n_atoms)
    
    # Weighted sum over atoms (multiply by charges and sum)
    S_real = torch.matmul(cos_Gr, charge)  # shape: (n_G,)
    S_imag = torch.matmul(sin_Gr, charge)  # shape: (n_G,)
    
    # |S(G)|^2 = S_real^2 + S_imag^2
    S2 = S_real**2 + S_imag**2
    
    # Compute energy contribution from each G
    # E_recip = sum_G [exp(-G^2/(4*alpha^2)) / G^2 * |S(G)|^2]
    prefactor = torch.exp(-G2 / (4 * alpha**2)) / G2
    E_recip = torch.sum(prefactor * S2)
    
    # Apply overall prefactor: (1/(2V)) * (1/(4π*ε0)) * 4π = 1/(2V*ε0)
    E_recip = E_recip / (2 * volume * epsilon_0)
    
    # Compute forces
    # F_i = -dE/dr_i
    # dE/dr_i = (1/(2V*ε0)) * sum_G [exp(-G²/(4α²))/G² * 2 * q_i * G * [sin(G·r_i)*S_real - cos(G·r_i)*S_imag]]
    
    # For each atom i and G vector: shape (n_G, n_atoms, 3)
    # Force contribution: prefactor[G] * 2 * q_i * G * [sin(G·r_i)*S_real - cos(G·r_i)*S_imag]
    
    # Compute [sin(G·r_i)*S_real - cos(G·r_i)*S_imag] for all i, G: shape (n_G, n_atoms)
    structure_deriv = sin_Gr * S_real.unsqueeze(1) - cos_Gr * S_imag.unsqueeze(1)
    
    # Multiply by charges: shape (n_G, n_atoms)
    structure_deriv = structure_deriv * charge.unsqueeze(0)
    
    # Multiply by 2 * prefactor: shape (n_G, n_atoms)
    force_coeff = 2 * prefactor.unsqueeze(1) * structure_deriv
    
    # Multiply by G vectors and sum over G: shape (n_atoms, 3)
    # force_coeff: (n_G, n_atoms), G_vectors: (n_G, 3)
    # We want for each atom: sum_G [force_coeff[G, i] * G_vectors[G]]
    forces = torch.matmul(force_coeff.T, G_vectors)  # (n_atoms, n_G) @ (n_G, 3) = (n_atoms, 3)
    
    # Apply overall prefactor and negative sign (force = -gradient)
    forces = -forces / (2 * volume * epsilon_0)
    
    return E_recip, forces

def potential_selfsum(r, charge, alpha):
    """self interaction sum"""
    E_self = -torch.sum((alpha / torch.sqrt(torch.pi)) * charge**2)
    return E_self, 0