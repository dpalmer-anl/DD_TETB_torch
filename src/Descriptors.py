import torch
import numpy as np


#try using Featomic. can get pairwise descriptors from this and equivariant atom centered descriptors from this.
def get_atom_centered_descriptors_featomic(atoms: ase.Atoms,cutoff: float) -> torch.Tensor:
    HYPERPARAMETERS = {
        "cutoff": {
            "radius": cutoff,
            "smoothing": {"type": "ShiftedCosine", "width": 0.5},
        },
        "density": {
            "type": "Gaussian",
            "width": 0.3,
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": 2,
            "radial": {"type": "Gto", "max_radial": 2},
        },
    }
    spex_calculator = SphericalExpansion(**HYPERPARAMETERS)
    descriptor_calculator = EquivariantPowerSpectrum(spex_calculator)
    power_spectrum = descriptor_calculator.compute(atoms)
    return power_spectrum

# Precomputed spherical Bessel zeros (l=0 to 9, first 10 zeros)
# Values are z_{ln} such that j_l(z_{ln}) = 0
BESSEL_ZEROS = {
    0: [3.14159265, 6.28318531, 9.42477796, 12.56637061, 15.70796327, 18.84955592, 21.99114858, 25.13274123, 28.27433388, 31.41592654],
    1: [4.49340946, 7.72525184, 10.90412166, 14.06619391, 17.22075527, 20.37130296, 23.5194525, 26.66605426, 29.81159879, 32.95638904],
    2: [5.7634592, 9.09501133, 12.32294097, 15.51460301, 18.68903636, 21.85387422, 25.0128032, 28.16782971, 31.32014171, 34.47048833],
    3: [6.987932, 10.41711855, 13.69802315, 16.92362129, 20.12180617, 23.30424699, 26.47676366, 29.64260454, 32.80373239, 35.9614058],
    4: [8.18256145, 11.70490715, 15.03966471, 18.30125596, 21.52541773, 24.72756555, 27.9155762, 31.09393321, 34.26539009, 37.43173677],
    5: [9.35581211, 12.96653017, 16.35470964, 19.6531521, 22.90455065, 26.12775014, 29.33256258, 32.52466129, 35.70757695, 38.88363096],
    6: [10.51283541, 14.20739246, 17.64797487, 20.98346307, 24.26276804, 27.50786836, 30.73038073, 33.9371083, 37.13233172, 40.31889251],
    7: [11.65703219, 15.43128921, 18.9229992, 22.29534802, 25.60285595, 28.87037335, 32.11119624, 35.33319418, 38.54136485, 41.73905287],
    8: [12.79078171, 16.64100288, 20.18247076, 23.59127482, 26.92704078, 30.21726271, 33.47680082, 36.71452913, 39.93612781, 43.14542502],
    9: [13.91582261, 17.8386432, 21.42848697, 24.87321392, 28.23713436, 31.55018838, 34.82869654, 38.08247909, 41.31786469, 44.53914463],
}


def get_disp(positions: torch.Tensor, cell: torch.Tensor, 
            i: torch.Tensor, j: torch.Tensor, 
            di: torch.Tensor, dj: torch.Tensor) -> torch.Tensor:
    """
    Get displacement vector between two atoms.
    """
    disp = di.unsqueeze(1) * cell[0] + \
           dj.unsqueeze(1) * cell[1] + \
           positions[j] - positions[i]
    
    return disp

def cutoff_function(dist: torch.Tensor, r_cut: float) -> torch.Tensor:
    """
    Cosine cutoff function. Adapted from Phys. Rev. B 100, 195419 (2019)
    
    Args:
        dist: (...,) tensor of distances between atoms
        r_cut: float cutoff radius
    
    Returns:
        (...,) tensor of cutoff values
    """
    mask = dist < r_cut
    result = torch.zeros_like(dist)
    result[mask] = 0.5 * (torch.cos(np.pi * dist[mask] / r_cut) + 1.0)
    return result

def compute_bond_orientation_descriptors(displacements: torch.Tensor,
                                        i_idx: torch.Tensor,
                                        j_idx: torch.Tensor,
                                        n_atoms: int,
                                        r_cut: float,
                                        sigma: float=1.42) -> torch.Tensor:

    distances = torch.norm(displacements,axis=1)
    bond_order = []
    bond_order_avg = []

    for i,disp in enumerate(displacements):
        use_ind = np.squeeze(np.where(i==i_idx))
        x = displacements[use_ind,0]
        y = displacements[use_ind,1]
        z = displacements[use_ind,2]
        theta = np.arctan(y/x)
        phi = np.arccos(z/np.linalg.norm(displacements[use_ind,:]))
        for l in range(max_l+1):
            bo = []
            for m in range(-l,l+1):
                val = np.mean(sph_harm_y(l,m,theta,phi)*np.exp(-distances[use_ind]**2/2/sigma**2))
                bond_order.append(val)

    return bond_order,bond_order_avg
    

    
def compute_radial_descriptors(dist_ij: torch.Tensor, 
                               i_idx: torch.Tensor,
                               n_atoms: int,
                               eta_vals: torch.Tensor, 
                               radial_centers: torch.Tensor,
                               r_cut: float) -> torch.Tensor:
    """
    Compute radial Behler symmetry functions (fully vectorized).
    G_radial_i = sum_{j≠i} exp(-eta*(r_ij - R_desc)^2) * fc(r_ij)
    
    Args:
        dist_ij: (Npairs,) tensor of distances
        i_idx: (Npairs,) tensor of center atom indices
        n_atoms: int number of atoms
        eta_vals: (n_radial,) tensor of eta hyperparameters
        radial_centers: (n_radial,) tensor of R_desc shift parameters
        r_cut: float cutoff radius
    
    Returns:
        (n_atoms, n_radial) tensor of radial descriptors for each atom
    """
    n_radial = len(eta_vals)
    
    # Compute cutoff function: (Npairs,)
    fc = cutoff_function(dist_ij, r_cut)
    
    # Broadcast: (Npairs, n_radial)
    dist_expanded = dist_ij.unsqueeze(1)  # (Npairs, 1)
    eta_expanded = eta_vals.unsqueeze(0)  # (1, n_radial)
    R_expanded = radial_centers.unsqueeze(0)  # (1, n_radial)
    
    # Compute Gaussian: (Npairs, n_radial)
    gaussian = torch.exp(-eta_expanded * (dist_expanded - R_expanded)**2)
    
    # Apply cutoff: (Npairs, n_radial)
    contributions = gaussian * fc.unsqueeze(1)
    
    # Sum over neighbors for each center atom
    radial_desc = torch.zeros(n_atoms, n_radial, dtype=dist_ij.dtype, device=dist_ij.device)
    radial_desc.index_add_(0, i_idx, contributions)
    
    return radial_desc


def compute_angular_descriptors(displacement_vector: torch.Tensor,
                                i_idx: torch.Tensor,
                                j_idx: torch.Tensor,
                                n_atoms: int,
                                lambda_vals: torch.Tensor,
                                ksi_vals: torch.Tensor,
                                eta_vals: torch.Tensor,
                                r_cut: float) -> torch.Tensor:
    """
    Compute angular Behler symmetry functions (fully vectorized).
    G_angular_i = 2^(1-ksi) * sum_{j≠i} sum_{k>j, k≠i} 
                  (1 + lambda*cos(theta_ijk))^ksi * 
                  exp(-eta*(r_ij^2 + r_ik^2 + r_jk^2)) * 
                  fc(r_ij) * fc(r_ik) * fc(r_jk)
    
    Args:
        displacement_vector: (Npairs, 3) tensor of displacement vectors
        i_idx: (Npairs,) tensor of center atom indices
        j_idx: (Npairs,) tensor of neighbor atom indices
        n_atoms: int number of atoms
        lambda_vals: (n_angular,) tensor of lambda hyperparameters
        ksi_vals: (n_angular,) tensor of ksi (zeta) hyperparameters
        eta_vals: (n_angular,) tensor of eta hyperparameters
        r_cut: float cutoff radius
    
    Returns:
        (n_atoms, n_angular) tensor of angular descriptors for each atom
    """
    n_angular = len(lambda_vals)
    device = displacement_vector.device
    dtype = displacement_vector.dtype
    n_pairs = len(displacement_vector)
    
    # Create all triplets (i, j, k) where both j and k are neighbors of i
    # Strategy: for each pair (i,j) at index idx_j and pair (i,k) at index idx_k,
    # create a triplet if they share the same center i and idx_k > idx_j
    
    # Expand to create all combinations of pairs
    # i_idx: (Npairs,), we need to find pairs that share the same center
    # Use broadcasting to create all pair combinations
    
    # Create masks for matching centers: (Npairs, Npairs)
    # This is memory intensive but fully vectorized
    same_center = i_idx.unsqueeze(1) == i_idx.unsqueeze(0)  # (Npairs, Npairs)
    
    # Create upper triangular mask to avoid double counting (k > j)
    idx_range = torch.arange(n_pairs, device=device)
    upper_triangular = idx_range.unsqueeze(1) < idx_range.unsqueeze(0)  # (Npairs, Npairs)
    
    # Combine masks: valid triplets where same center and k > j
    valid_triplets = same_center & upper_triangular  # (Npairs, Npairs)
    
    # Get indices of valid triplets
    idx_j_triplet, idx_k_triplet = torch.where(valid_triplets)  # Each has shape (n_triplets,)
    
    if len(idx_j_triplet) == 0:
        # No valid triplets
        return torch.zeros(n_atoms, n_angular, dtype=dtype, device=device)
    
    # Extract triplet information
    i_triplet = i_idx[idx_j_triplet]  # (n_triplets,) - center atoms
    j_triplet = j_idx[idx_j_triplet]  # (n_triplets,) - first neighbor
    k_triplet = j_idx[idx_k_triplet]  # (n_triplets,) - second neighbor
    
    r_ij = displacement_vector[idx_j_triplet]  # (n_triplets, 3)
    r_ik = displacement_vector[idx_k_triplet]  # (n_triplets, 3)
    r_jk = r_ik - r_ij  # (n_triplets, 3)
    
    # Compute distances
    dist_ij = torch.norm(r_ij, dim=1)  # (n_triplets,)
    dist_ik = torch.norm(r_ik, dim=1)  # (n_triplets,)
    dist_jk = torch.norm(r_jk, dim=1)  # (n_triplets,)
    
    # Compute cutoff functions
    fc_ij = cutoff_function(dist_ij, r_cut)  # (n_triplets,)
    fc_ik = cutoff_function(dist_ik, r_cut)  # (n_triplets,)
    fc_jk = cutoff_function(dist_jk, r_cut)  # (n_triplets,)
    
    # Compute angles: cos(theta_ijk) = r_ij · r_ik / (|r_ij| * |r_ik|)
    cos_theta = torch.sum(r_ij * r_ik, dim=1) / (dist_ij * dist_ik + 1e-8)  # (n_triplets,)
    
    # Broadcast for all hyperparameters: (n_triplets, n_angular)
    cos_theta_exp = cos_theta.unsqueeze(1)  # (n_triplets, 1)
    lambda_exp = lambda_vals.unsqueeze(0)  # (1, n_angular)
    ksi_exp = ksi_vals.unsqueeze(0)  # (1, n_angular)
    eta_exp = eta_vals.unsqueeze(0)  # (1, n_angular)
    
    # Compute angular term: (1 + lambda*cos(theta))^ksi
    angular_term = (1 + lambda_exp * cos_theta_exp)**ksi_exp  # (n_triplets, n_angular)
    
    # Compute Gaussian term: exp(-eta*(r_ij^2 + r_ik^2 + r_jk^2))
    dist_sum = (dist_ij**2 + dist_ik**2 + dist_jk**2).unsqueeze(1)  # (n_triplets, 1)
    gaussian_term = torch.exp(-eta_exp * dist_sum)  # (n_triplets, n_angular)
    
    # Prefactor: 2^(1-ksi)
    prefactor = 2**(1 - ksi_exp)  # (1, n_angular)
    
    # Cutoff product
    fc_product = (fc_ij * fc_ik * fc_jk).unsqueeze(1)  # (n_triplets, 1)
    
    # Total contribution: (n_triplets, n_angular)
    contributions = prefactor * angular_term * gaussian_term * fc_product
    
    # Accumulate contributions for each center atom
    angular_desc = torch.zeros(n_atoms, n_angular, dtype=dtype, device=device)
    angular_desc.index_add_(0, i_triplet, contributions)
    
    return angular_desc


def compute_angular_descriptors_pairwise(displacement_vector: torch.Tensor,
                                        i_idx: torch.Tensor,
                                        j_idx: torch.Tensor,
                                        lambda_vals: torch.Tensor,
                                        ksi_vals: torch.Tensor,
                                        eta_vals: torch.Tensor,
                                        r_cut: float) -> torch.Tensor:
    """
    Compute angular descriptors for pairs (i,j), contracted only over k sum (fully vectorized).
    This is for pairwise descriptors.
    
    For each pair (i, j), compute:
    G_angular_ij = 2^(1-ksi) * sum_{k≠i,j} 
                   (1 + lambda*cos(theta_ijk))^ksi * 
                   exp(-eta*(r_ij^2 + r_ik^2 + r_jk^2)) * 
                   fc(r_ij) * fc(r_ik) * fc(r_jk)
    
    Args:
        displacement_vector: (Npairs, 3) tensor of displacement vectors
        i_idx: (Npairs,) tensor of center atom indices
        j_idx: (Npairs,) tensor of neighbor atom indices
        lambda_vals: (n_angular,) tensor of lambda hyperparameters
        ksi_vals: (n_angular,) tensor of ksi hyperparameters
        eta_vals: (n_angular,) tensor of eta hyperparameters
        r_cut: float cutoff radius
    
    Returns:
        (Npairs, n_angular) tensor of angular descriptors for each pair
    """
    n_angular = len(lambda_vals)
    n_pairs = len(displacement_vector)
    device = displacement_vector.device
    dtype = displacement_vector.dtype
    
    # For each pair (i,j) at index pair_ij, find all k where (i,k) exists
    # Create mapping: for each pair index, find other pairs with same center i
    
    # same_center[pair_ij, pair_ik] = True if both have same center i
    same_center = i_idx.unsqueeze(1) == i_idx.unsqueeze(0)  # (Npairs, Npairs)
    
    # different_neighbor[pair_ij, pair_ik] = True if they have different neighbors
    different_neighbor = j_idx.unsqueeze(1) != j_idx.unsqueeze(0)  # (Npairs, Npairs)
    
    # Valid combinations: same center i, different neighbors j≠k
    valid_combos = same_center & different_neighbor  # (Npairs, Npairs)
    
    # For each pair_ij, find indices of valid pair_ik
    pair_ij_idx, pair_ik_idx = torch.where(valid_combos)  # Each has shape (n_combos,)
    
    if len(pair_ij_idx) == 0:
        # No valid combinations
        return torch.zeros(n_pairs, n_angular, dtype=dtype, device=device)
    
    # Extract information for each combination
    i_combo = i_idx[pair_ij_idx]  # (n_combos,) - should all be same as i_idx[pair_ik_idx]
    j_combo = j_idx[pair_ij_idx]  # (n_combos,)
    k_combo = j_idx[pair_ik_idx]  # (n_combos,)
    
    r_ij = displacement_vector[pair_ij_idx]  # (n_combos, 3)
    r_ik = displacement_vector[pair_ik_idx]  # (n_combos, 3)
    r_jk = r_ik - r_ij  # (n_combos, 3)
    
    # Compute distances
    dist_ij = torch.norm(r_ij, dim=1)  # (n_combos,)
    dist_ik = torch.norm(r_ik, dim=1)  # (n_combos,)
    dist_jk = torch.norm(r_jk, dim=1)  # (n_combos,)
    
    # Compute cutoff functions
    fc_ij = cutoff_function(dist_ij, r_cut)  # (n_combos,)
    fc_ik = cutoff_function(dist_ik, r_cut)  # (n_combos,)
    fc_jk = cutoff_function(dist_jk, r_cut)  # (n_combos,)
    
    # Compute angles
    cos_theta = torch.sum(r_ij * r_ik, dim=1) / (dist_ij * dist_ik + 1e-8)  # (n_combos,)
    
    # Broadcast for hyperparameters: (n_combos, n_angular)
    cos_theta_exp = cos_theta.unsqueeze(1)  # (n_combos, 1)
    lambda_exp = lambda_vals.unsqueeze(0)  # (1, n_angular)
    ksi_exp = ksi_vals.unsqueeze(0)  # (1, n_angular)
    eta_exp = eta_vals.unsqueeze(0)  # (1, n_angular)
    
    # Compute angular term
    angular_term = (1 + lambda_exp * cos_theta_exp)**ksi_exp  # (n_combos, n_angular)
    
    # Compute Gaussian term
    dist_sum = (dist_ij**2 + dist_ik**2 + dist_jk**2).unsqueeze(1)  # (n_combos, 1)
    gaussian_term = torch.exp(-eta_exp * dist_sum)  # (n_combos, n_angular)
    
    # Prefactor
    prefactor = 2**(1 - ksi_exp)  # (1, n_angular)
    
    # Cutoff product
    fc_product = (fc_ij * fc_ik * fc_jk).unsqueeze(1)  # (n_combos, 1)
    
    # Total contribution: (n_combos, n_angular)
    contributions = prefactor * angular_term * gaussian_term * fc_product
    
    # Accumulate contributions for each pair (i,j)
    angular_desc_pairs = torch.zeros(n_pairs, n_angular, dtype=dtype, device=device)
    angular_desc_pairs.index_add_(0, pair_ij_idx, contributions)
    
    return angular_desc_pairs


def get_atom_centered_descriptors(displacement_vector: torch.Tensor, 
                                  i_idx: torch.Tensor, 
                                  j_idx: torch.Tensor,
                                  n_atoms: int,
                                  eta_radial: torch.Tensor, 
                                  radial_centers: torch.Tensor,
                                  lambda_vals: torch.Tensor, 
                                  ksi_vals: torch.Tensor,
                                  eta_angular: torch.Tensor,
                                  r_cut: float) -> torch.Tensor:
    """
    Get atom-centered descriptors combining radial and angular terms.
    
    Args:
        displacement_vector: (Npairs, 3) tensor of displacement vectors
        i_idx: (Npairs,) tensor of center atom indices
        j_idx: (Npairs,) tensor of neighbor atom indices
        n_atoms: int number of atoms
        eta_radial: (n_radial,) tensor of eta parameters for radial descriptors
        radial_centers: (n_radial,) tensor of R_desc shift parameters
        lambda_vals: (n_angular,) tensor of lambda parameters
        ksi_vals: (n_angular,) tensor of ksi parameters
        eta_angular: (n_angular,) tensor of eta parameters for angular descriptors
        r_cut: float cutoff radius
    
    Returns:
        (n_atoms, n_radial + n_angular) tensor of atom-centered descriptors
    """
    # Compute distances
    dist_ij = torch.norm(displacement_vector, dim=1)
    
    # Compute radial descriptors
    radial_desc = compute_radial_descriptors(dist_ij, i_idx, n_atoms, 
                                            eta_radial, radial_centers, r_cut)
    
    # Compute angular descriptors
    angular_desc = compute_angular_descriptors(displacement_vector, i_idx, j_idx, n_atoms,
                                              lambda_vals, ksi_vals, eta_angular, r_cut)
    
    # Concatenate
    descriptors = torch.cat([radial_desc, angular_desc], dim=1)
    
    return descriptors


def get_pairwise_descriptors(displacement_vector: torch.Tensor, 
                             i_idx: torch.Tensor, 
                             j_idx: torch.Tensor,
                             lambda_vals: torch.Tensor, 
                             ksi_vals: torch.Tensor,
                             eta_angular: torch.Tensor,
                             r_cut: float) -> torch.Tensor:
    """
    Get pairwise descriptors: interatomic distance + angular descriptors (contracted over k).
    Ignores radial descriptors as requested.
    
    Args:
        displacement_vector: (Npairs, 3) tensor of displacement vectors
        i_idx: (Npairs,) tensor of center atom indices
        j_idx: (Npairs,) tensor of neighbor atom indices
        lambda_vals: (n_angular,) tensor of lambda parameters
        ksi_vals: (n_angular,) tensor of ksi parameters
        eta_angular: (n_angular,) tensor of eta parameters
        r_cut: float cutoff radius
    
    Returns:
        (Npairs, 1 + n_angular) tensor of pairwise descriptors
    """
    # Compute distances
    dist_ij = torch.norm(displacement_vector, dim=1, keepdim=True)  # (Npairs, 1)
    
    # Apply cutoff to distance
    fc = cutoff_function(dist_ij.squeeze(1), r_cut).unsqueeze(1)
    dist_smoothed = dist_ij * fc
    
    # Compute angular descriptors for pairs
    angular_desc_pairs = compute_angular_descriptors_pairwise(
        displacement_vector, i_idx, j_idx, 
        lambda_vals, ksi_vals, eta_angular, r_cut)
    
    # Concatenate distance with angular descriptors
    descriptors = torch.cat([dist_smoothed, angular_desc_pairs], dim=1)
    
    return descriptors

def get_pairwise_descpriptors_ace(atom_centered_descriptors: torch.Tensor,
                                  i_idx: torch.Tensor,
                                  j_idx: torch.Tensor,
                                  distances: torch.Tensor,
                                  lambda_vals: torch.Tensor) -> torch.Tensor:
    """
    Get pairwise descriptors: interatomic distance + angular descriptors (contracted over k).
    Ignores radial descriptors as requested.
    
    Args:
        atom_centered_descriptors: (n_atoms, n_radial + n_angular) tensor of atom-centered descriptors
        distances: (Npairs, 1) tensor of distances
        lambda_vals: (n_angular,) tensor of lambda parameters
    
    Returns:
        (Npairs, n_radial +n_angular) tensor of pairwise descriptors
    """
    n_angular = len(lambda_vals)
    n_atoms = atom_centered_descriptors.shape[0]
    device = atom_centered_descriptors.device
    pairwise_descriptors = torch.exp(- distances/lambda_vals ) * atom_centered_descriptors[i_idx, :] * atom_centered_descriptors[j_idx, :]
    return pairwise_descriptors


def get_charge_weighted_descriptors(displacement_vector: torch.Tensor, 
                                    charge: torch.Tensor,
                                    i_idx: torch.Tensor, 
                                    j_idx: torch.Tensor,
                                    n_atoms: int,
                                    eta_radial: torch.Tensor, 
                                    radial_centers: torch.Tensor,
                                    lambda_vals: torch.Tensor, 
                                    ksi_vals: torch.Tensor,
                                    eta_angular: torch.Tensor,
                                    r_cut: float) -> torch.Tensor:
    """
    Get charge-weighted descriptors: same as atom-centered but weighted by charge at each neighbor.
    
    For radial: G_i = sum_j q_j * exp(-eta*(r_ij - R_desc)^2) * fc(r_ij)
    For angular: similar weighting applies
    
    Args:
        displacement_vector: (Npairs, 3) tensor of displacement vectors
        charge: (n_atoms,) tensor of charges on each atom
        i_idx: (Npairs,) tensor of center atom indices
        j_idx: (Npairs,) tensor of neighbor atom indices
        n_atoms: int number of atoms
        eta_radial: (n_radial,) tensor of eta parameters for radial descriptors
        radial_centers: (n_radial,) tensor of R_desc shift parameters
        lambda_vals: (n_angular,) tensor of lambda parameters
        ksi_vals: (n_angular,) tensor of ksi parameters
        eta_angular: (n_angular,) tensor of eta parameters for angular descriptors
        r_cut: float cutoff radius
    
    Returns:
        (n_atoms, n_radial + n_angular) tensor of charge-weighted descriptors
    """
    # Compute distances
    dist_ij = torch.norm(displacement_vector, dim=1)
    fc = cutoff_function(dist_ij, r_cut)
    
    n_radial = len(eta_radial)
    n_angular = len(lambda_vals)
    
    # Weighted radial descriptors (fully vectorized)
    dist_expanded = dist_ij.unsqueeze(1)  # (Npairs, 1)
    eta_expanded = eta_radial.unsqueeze(0)  # (1, n_radial)
    R_expanded = radial_centers.unsqueeze(0)  # (1, n_radial)
    
    gaussian = torch.exp(-eta_expanded * (dist_expanded - R_expanded)**2)
    
    # Weight by charge on neighbor j
    q_weights = charge[j_idx].unsqueeze(1)  # (Npairs, 1)
    contributions = gaussian * fc.unsqueeze(1) * q_weights
    
    radial_desc_weighted = torch.zeros(n_atoms, n_radial, 
                                       dtype=displacement_vector.dtype, 
                                       device=displacement_vector.device)
    radial_desc_weighted.index_add_(0, i_idx, contributions)
    
    # Angular descriptors (use standard computation)
    angular_desc = compute_angular_descriptors(displacement_vector, i_idx, j_idx, n_atoms,
                                              lambda_vals, ksi_vals, eta_angular, r_cut)
    
    # Combine
    descriptors_weighted = torch.cat([radial_desc_weighted, angular_desc], dim=1)
    
    return descriptors_weighted


def _spherical_bessel_jn(l: int, x: torch.Tensor) -> torch.Tensor:
    """
    Compute spherical Bessel function j_l(x) recursively.
    """
    if l == 0:
        # j0(x) = sin(x)/x
        # Handle x=0 case carefully
        return torch.where(x < 1e-8, torch.ones_like(x), torch.sin(x) / x)
    elif l == 1:
        # j1(x) = sin(x)/x^2 - cos(x)/x
        mask = x < 1e-8
        val = torch.where(mask, torch.zeros_like(x), 
                          torch.sin(x) / (x**2) - torch.cos(x) / x)
        # Limit x->0 for j1 is 0
        return val
    else:
        # Recurrence: j_{l+1} = (2l+1)/x * j_l - j_{l-1}
        j_lminus1 = _spherical_bessel_jn(0, x)
        j_l = _spherical_bessel_jn(1, x)
        
        for i in range(1, l):
            j_next = ((2*i + 1) / x) * j_l - j_lminus1
            j_lminus1 = j_l
            j_l = j_next
            
        return j_l

def _compute_real_spherical_harmonics(vectors: torch.Tensor, l_max: int) -> torch.Tensor:
    """
    Compute real spherical harmonics Y_lm for a batch of vectors.
    Returns tensor of shape (N, (l_max+1)^2).
    
    vectors: (N, 3)
    """
    n_vecs = vectors.shape[0]
    device = vectors.device
    dtype = vectors.dtype
    
    # Normalize vectors to unit sphere
    r = torch.norm(vectors, dim=1, keepdim=True)
    # Handle zero vectors
    mask = r.squeeze() < 1e-8
    unit_vecs = torch.zeros_like(vectors)
    unit_vecs[~mask] = vectors[~mask] / r[~mask]
    
    x = unit_vecs[:, 0]
    y = unit_vecs[:, 1]
    z = unit_vecs[:, 2]
    
    # cos(theta) = z
    costheta = z
    
    # phi
    # We need sin(m*phi) and cos(m*phi).
    # x = sin(theta)cos(phi), y = sin(theta)sin(phi)
    # planar radius R = sqrt(x^2+y^2) = sin(theta)
    R = torch.sqrt(x**2 + y**2)
    
    # Avoid division by zero for phi
    # If R is 0, phi is undefined, but Y_lm terms with m!=0 are 0 anyway.
    # We can set cosphi=1, sinphi=0 arbitrarily.
    mask_pole = R < 1e-8
    cosphi = torch.zeros_like(x)
    sinphi = torch.zeros_like(x)
    cosphi[~mask_pole] = x[~mask_pole] / R[~mask_pole]
    sinphi[~mask_pole] = y[~mask_pole] / R[~mask_pole]
    cosphi[mask_pole] = 1.0
    
    # Precompute cos(m*phi) and sin(m*phi)
    cos_m_phi = [torch.ones_like(x), cosphi]
    sin_m_phi = [torch.zeros_like(x), sinphi]
    
    for m in range(2, l_max + 1):
        # cos(m phi) = cos((m-1)phi + phi) = cos((m-1)phi)cos(phi) - sin((m-1)phi)sin(phi)
        cmp = cos_m_phi[-1] * cosphi - sin_m_phi[-1] * sinphi
        smp = sin_m_phi[-1] * cosphi + cos_m_phi[-1] * sinphi
        cos_m_phi.append(cmp)
        sin_m_phi.append(smp)
        
    # Implement Associated Legendre Polynomials P_lm(x)
    # Recursive approach or direct formulas?
    # For arbitrary l_max, recursion is best.
    
    # We will store Y_lm values in a list and then stack
    Y_lm_list = []
    
    # Scaling factors for normalization can be precomputed or computed on fly
    # Y_lm = N_lm * P_lm(cos theta) * {cos(m phi) or sin(m phi)}
    
    # Iterate l from 0 to l_max
    # We need P_mm, P_m+1,m ...
    
    # P_mm(x) = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
    # where x = cos(theta), so (1-x^2)^(1/2) = sin(theta) = R
    
    # Precompute factorials or prefactors if needed
    
    # Just implement full recursion for Y_lm directly if possible?
    # Or stick to P_lm.
    
    # Let's use a simpler dictionary approach for P_lm values
    P = {}  # (l, m) -> tensor
    
    # P_00 = 1
    P[(0, 0)] = torch.ones_like(z)
    
    for l in range(1, l_max + 1):
        # Compute P_ll first
        # P_ll = -(2l-1) * sin(theta) * P_{l-1, l-1}
        # Note: The standard definition includes (-1)^m phase.
        # (2l-1)!! = (2l-1) * (2l-3) ...
        
        P[(l, l)] = - (2 * l - 1) * R * P[(l-1, l-1)]
        
        # Compute P_{l, l-1}
        # P_{l, l-1} = x * (2l-1) * P_{l-1, l-1}
        if l-1 >= 0:
            P[(l, l-1)] = z * (2 * l - 1) * P[(l-1, l-1)]
        
        # Compute remaining P_{l, m} for m < l-1
        # P_{l, m} = ((2l-1) * x * P_{l-1, m} - (l+m-1) * P_{l-2, m}) / (l-m)
        for m in range(0, l-1):
            P[(l, m)] = ((2 * l - 1) * z * P[(l-1, m)] - (l + m - 1) * P[(l-2, m)]) / (l - m)

    # Now construct Y_lm
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Normalization constant
            # N_lm = sqrt( (2l+1)/(4pi) * (l-|m|)! / (l+|m|)! )
            
            # Factorials
            fact_diff = float(np.math.factorial(l - abs(m)))
            fact_sum = float(np.math.factorial(l + abs(m)))
            prefactor = np.sqrt((2 * l + 1) / (4 * np.pi) * fact_diff / fact_sum)
            
            p_val = P[(l, abs(m))]
            
            if m == 0:
                y = prefactor * p_val
            elif m > 0:
                y = np.sqrt(2) * prefactor * p_val * cos_m_phi[m]
            else: # m < 0
                y = np.sqrt(2) * prefactor * p_val * sin_m_phi[abs(m)]
            
            Y_lm_list.append(y)
            
    return torch.stack(Y_lm_list, dim=1)

def get_atom_centered_descriptors_ace(displacement_vector: torch.Tensor, 
                                      i_idx: torch.Tensor, 
                                      j_idx: torch.Tensor,
                                      n_atoms: int,
                                      n_max: int, 
                                      l_max: int,
                                      r_cut: float) -> torch.Tensor:
    """
    Calculate ACE (Atomic Cluster Expansion) descriptors for arbitrary moments using 
    Bessel functions for the radial basis.
    
    Specifically, this computes the Power Spectrum (SOAP-like invariants), 
    which corresponds to the 2-body correlation of the atomic density expansion.
    
    The atomic density is expanded as:
    A_{nlm} = sum_j R_{nl}(r_{ij}) * Y_{lm}(hat{r}_{ij})
    
    where R_{nl} are spherical Bessel functions constrained to be zero at r_cut.
    
    The descriptors returned are the rotational invariants (Power Spectrum):
    P_{nn'l} = sum_m A_{nlm} * A_{n'lm}
    
    Args:
        displacement_vector: (Npairs, 3) tensor of displacement vectors
        i_idx: (Npairs,) tensor of center atom indices
        j_idx: (Npairs,) tensor of neighbor atom indices
        n_atoms: int number of atoms
        n_max: int maximum radial order (0 to n_max)
        l_max: int maximum angular order (0 to l_max)
        r_cut: float cutoff radius
        
    Returns:
        descriptors: (n_atoms, (n_max+1)^2 * (l_max+1)) tensor of ACE descriptors
    """
    device = displacement_vector.device
    dtype = displacement_vector.dtype
    
    # 1. Compute distances and unit vectors
    dist_ij = torch.norm(displacement_vector, dim=1)
    
    # Cutoff function
    fc = cutoff_function(dist_ij, r_cut)
    
    # 2. Compute Radial Basis R_{nl}
    # We use spherical Bessel functions j_l(z_{nl} * r / r_cut)
    # z_{nl} is the n-th zero of j_l
    
    R_nl_list = []
    # To fully vectorize, we compute R_{nl} for all n, l
    
    for l in range(l_max + 1):
        # Use precomputed Bessel zeros
        if l in BESSEL_ZEROS:
            zeros_list = BESSEL_ZEROS[l]
            if n_max + 1 > len(zeros_list):
                raise ValueError(f"n_max={n_max} exceeds available precomputed Bessel zeros ({len(zeros_list)})")
            zeros = torch.tensor(zeros_list[:n_max+1], device=device, dtype=dtype)
        else:
            raise ValueError(f"l={l} not supported in precomputed Bessel zeros (max l=9)")
            
        for n in range(n_max + 1):
            z_nl = zeros[n]
            
            # Argument for Bessel function
            x = z_nl * dist_ij / r_cut
            
            # Compute j_l(x)
            # Use our recursive implementation or map to built-in if available
            # For x > z_nl (r > r_cut), this might oscillate, but we apply cutoff fc
            bessel_val = _spherical_bessel_jn(l, x)
            
            # Normalization (optional but recommended)
            # int_0^a r^2 j_l(z_{nl} r/a)^2 dr = a^3/2 * j_{l+1}(z_{nl})^2
            # We want orthonormal basis in radial direction?
            # Let's just use the raw basis multiplied by fc for smoothness
            
            # Apply cutoff function to radial part
            # Ideally the basis naturally decays, but we enforce smooth cutoff
            R_nl = bessel_val * fc
            
            R_nl_list.append(R_nl) # Flattened list of R_{nl}
            
    # Stack R_nl: (Npairs, (l_max+1)*(n_max+1))
    R_nl_all = torch.stack(R_nl_list, dim=1)
    
    # 3. Compute Spherical Harmonics Y_{lm}
    # (Npairs, (l_max+1)^2)
    Y_lm_all = _compute_real_spherical_harmonics(displacement_vector, l_max)
    
    # 4. Compute Atomic Base A_{nlm}
    # A_{nlm} = sum_j R_{nl}(r_j) * Y_{lm}(r_j)
    # We need to form the product for valid combinations
    # Indices:
    # R_nl is indexed by (l, n) flattened
    # Y_lm is indexed by (l, m) flattened
    
    # We can do this block by block per l to save memory and logic
    
    # Tensor to store A_{nlm} for each atom
    # Structure: List of tensors or a large tensor?
    # Let's store A_{nlm} in a structured way: (n_atoms, n_max+1, (l_max+1)^2)
    # Wait, Y_lm depends on l. R_nl depends on n and l.
    # So for a fixed l, we have (n_max+1) radial functions and (2l+1) angular functions.
    
    # We will compute invariants block by block for each l
    invariants_list = []
    
    # Offset in R_nl_all for current l
    r_offset = 0
    # Offset in Y_lm_all for current l
    y_offset = 0
    
    for l in range(l_max + 1):
        n_radial_l = n_max + 1
        n_angular_l = 2 * l + 1
        
        # Extract R_{nl} for this l: (Npairs, n_max+1)
        R_l = R_nl_all[:, r_offset : r_offset + n_radial_l]
        r_offset += n_radial_l
        
        # Extract Y_{lm} for this l: (Npairs, 2l+1)
        Y_l = Y_lm_all[:, y_offset : y_offset + n_angular_l]
        y_offset += n_angular_l
        
        # Compute product phi_{nlm} = R_{nl} * Y_{lm}
        # Shape: (Npairs, n_max+1, 2l+1)
        # Use broadcasting
        phi_l = R_l.unsqueeze(2) * Y_l.unsqueeze(1)
        
        # Flatten to sum: (Npairs, (n_max+1)*(2l+1))
        phi_l_flat = phi_l.reshape(len(dist_ij), -1)
        
        # Sum over neighbors to get A_{nlm} for this l
        A_l_flat = torch.zeros(n_atoms, phi_l_flat.shape[1], device=device, dtype=dtype)
        A_l_flat.index_add_(0, i_idx, phi_l_flat)
        
        # Reshape back: (n_atoms, n_max+1, 2l+1)
        A_l = A_l_flat.reshape(n_atoms, n_max + 1, 2 * l + 1)
        
        # 5. Compute Power Spectrum P_{nn'l}
        # P_{nn'l} = sum_m A_{nlm} * A_{n'lm}
        # Matrix multiplication over m
        # (N, n, m) @ (N, m, n') -> (N, n, n')
        
        # A_l: (N, n, m)
        # P_l: (N, n, n)
        P_l = torch.bmm(A_l, A_l.transpose(1, 2))
        
        # Flatten P_l: (N, (n_max+1)^2)
        P_l_flat = P_l.reshape(n_atoms, -1)
        
        invariants_list.append(P_l_flat)
        
    # Concatenate all invariants
    # Total size: sum_l (n_max+1)^2 = (l_max+1) * (n_max+1)^2
    descriptors = torch.cat(invariants_list, dim=1)
    
    return descriptors

if __name__ =="__main__":
    from ase.build import graphene
    from NeighborList import get_neighbor_list
    atoms = graphene(vacuum=10.0)
    charge = torch.ones(len(atoms))
    r_cut = 2.5 # first and second nearest neighbors, each atom has 9 neighbors
    displacement_vector, i_idx, j_idx, di, dj = get_neighbor_list(torch.tensor(atoms.positions), torch.tensor(atoms.cell),rcut=r_cut)
    print("Number of pairs:", len(i_idx))
    print("Number of atoms:", len(atoms))
    print("distances: ",torch.norm(displacement_vector, dim=1))

    eta_radial = torch.tensor([1.0, 2.0, 3.0])
    radial_centers = torch.tensor([0.5, 1.0, 1.5])
    lambda_vals = torch.tensor([1.0, 2.0, 3.0])
    ksi_vals = torch.tensor([1.0, 2.0, 3.0])
    eta_angular = torch.tensor([1.0, 2.0, 3.0])
    
    atom_centered_descriptors = get_atom_centered_descriptors(displacement_vector, i_idx, j_idx, len(atoms), 
                                    eta_radial, radial_centers, lambda_vals, ksi_vals, eta_angular, r_cut)
    print("atom centered descriptors shape:", atom_centered_descriptors.shape)

    pairwise_descriptors = get_pairwise_descriptors(displacement_vector, i_idx, j_idx, lambda_vals, ksi_vals, eta_angular, r_cut)
    print("pairwise descriptors shape:", pairwise_descriptors.shape)

    charge_weighted_descriptors = get_charge_weighted_descriptors(displacement_vector, charge, i_idx, j_idx, len(atoms), 
                                    eta_radial, radial_centers, lambda_vals, ksi_vals, eta_angular, r_cut)
    print("charge weighted descriptors shape:", charge_weighted_descriptors.shape)
    
    # Test ACE descriptors
    n_max = 4
    l_max = 3
    ace_desc = get_atom_centered_descriptors_ace(displacement_vector, i_idx, j_idx, len(atoms), 
                                                n_max, l_max, r_cut)
    print("ACE descriptors shape:", ace_desc.shape)
    expected_dim = (n_max + 1)**2 * (l_max + 1)
    print(f"Expected dimension: {expected_dim}")
