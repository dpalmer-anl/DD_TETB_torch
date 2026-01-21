import torch

def cdist_torch(XA: torch.Tensor, XB: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-optimized pairwise Euclidean distances with Intel GPU acceleration.
    Equivalent to scipy.spatial.distance.cdist(XA, XB, metric='euclidean').
    """
    XA_norm = torch.sum(XA**2, dim=1, keepdim=True)   # shape (m, 1)
    XB_norm = torch.sum(XB**2, dim=1, keepdim=True).T   # shape (1, n)
    cross = torch.mm(XA, XB.T)                           # shape (m, n)
    D2 = XA_norm + XB_norm - 2 * cross
    return torch.sqrt(torch.clamp(D2, min=0.0))
    
def get_neighbor_list(positions: torch.Tensor, cell: torch.Tensor,  
                   rcut: float = 6.0, ):
    """
    PyTorch-optimized displacement calculation with Intel GPU acceleration.
    
    Args:
        positions: Atomic positions (N, 3)
        cell: Unit cell vectors (3, 3)
        cutoff: Distance cutoff
    
    Returns:
        disp, i, j, di, dj
    """
    #positions = positions.to(device=device, dtype=torch.float32)
    #cell = cell.to(device=device, dtype=torch.float32)
    
    natoms = len(positions)
    
    # Create extended coordinates
    di_list = []
    dj_list = []
    extended_coords = []
    
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            displaced_positions = positions + cell[0] * dx + cell[1] * dy
            extended_coords.append(displaced_positions)
            di_list.extend([dx] * natoms)
            dj_list.extend([dy] * natoms)
    
    extended_coords = torch.cat(extended_coords, dim=0)
    di = torch.tensor(di_list)
    dj = torch.tensor(dj_list)
    
    # Calculate distances
    distances = cdist_torch(positions, extended_coords)
    
    # Find valid pairs
    valid_mask = (distances > 0.529) & (distances < rcut)
    i, j = torch.where(valid_mask)
    
    di_valid = di[j]
    dj_valid = dj[j]
    j_valid = j % natoms
    
    # Calculate displacements
    disp = di_valid.unsqueeze(1) * cell[0] + \
           dj_valid.unsqueeze(1) * cell[1] + \
           positions[j_valid] - positions[i]
    
    return disp, i, j_valid, di_valid, dj_valid

def get_neighbor_list_deprecated(
    positions: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-friendly 2D displacement computation using radius_graph with periodic boundary conditions.

    Args:
        positions: (N, 3) tensor of atomic positions (Cartesian)
        cell: (3, 3) tensor of unit cell vectors (only a1, a2 relevant for 2D)
        atom_types: (N,) tensor of integer atom types
        cutoff: distance cutoff [Angstrom]
        type: "all", "intralayer", or "interlayer"

    Returns:
        disp: (M, 3) displacement vectors with PBC
        i, j: (M,) atom index pairs
        di, dj: (M,) integer image shifts along a1 and a2
    """
    """
    Compute 2D periodic displacements (matching original get_disp)
    using torch_cluster.radius. Self pairs are excluded.
    """
    device = positions.device
    natoms = positions.shape[0]

    # --- build periodic images in 2D ---
    shifts = torch.tensor([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)],
                          device=device, dtype=torch.long)
    all_pos_list, di_list, dj_list = [], [], []
    for s in shifts:
        all_pos_list.append(positions + s[0].float() * cell[0] + s[1].float() * cell[1])
        di_list.append(torch.full((natoms,), s[0], dtype=torch.long, device=device))
        dj_list.append(torch.full((natoms,), s[1], dtype=torch.long, device=device))

    all_pos = torch.cat(all_pos_list, dim=0)
    di_all = torch.cat(di_list, dim=0)
    dj_all = torch.cat(dj_list, dim=0)

    # --- radius search between central cell (y) and all images (x) ---
    idx_a, idx_b = radius(x=all_pos, y=positions, r=cutoff, max_num_neighbors=64)

    # determine which is which (neighbor vs center)
    max_a = int(idx_a.max().item())
    max_b = int(idx_b.max().item())
    if max_a >= natoms and max_b < natoms:
        neighbor_idx, center_idx = idx_a, idx_b
    elif max_b >= natoms and max_a < natoms:
        neighbor_idx, center_idx = idx_b, idx_a
    else:
        neighbor_idx, center_idx = (idx_a, idx_b) if max_a >= max_b else (idx_b, idx_a)

    # map neighbor to base atom
    j_base = neighbor_idx % natoms
    di_valid = di_all[neighbor_idx]
    dj_valid = dj_all[neighbor_idx]
    i_idx = center_idx

    # compute displacements
    disp = (
        di_valid.to(positions.dtype).unsqueeze(1) * cell[0]
        + dj_valid.to(positions.dtype).unsqueeze(1) * cell[1]
        + positions[j_base]
        - positions[i_idx]
    )

    # --- filter out self pairs (zero displacement and no lattice shift) ---
    not_self = ~((i_idx == j_base) & (di_valid == 0) & (dj_valid == 0))
    disp = disp[not_self]
    i_idx = i_idx[not_self]
    j_base = j_base[not_self]
    di_valid = di_valid[not_self]
    dj_valid = dj_valid[not_self]

    return disp, i_idx, j_base, di_valid, dj_valid