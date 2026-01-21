import numpy as np
import torch
from ase import Atoms
from ase.build import graphene
from Descriptors import get_disp
from Model import DD_TETB_model
from TB_Utils import create_hopping_function, k_path, k_uniform_mesh
import flatgraphene as fg
import matplotlib.pyplot as plt
import os

def generate_cubic_mesh(nx=2, ny=2, nz=2, spacing=1.42):
    """
    Generate Nx x Ny x Nz atoms on a simple cubic 3D grid.
    spacing: lattice constant / atom separation (Å)
    Returns ASE Atoms object and positions array.
    """
    # Cartesian coordinates of all atoms
    coords = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                coords.append([i * spacing, j * spacing, k * spacing])
    coords = np.array(coords) + np.random.uniform(0, 0.02, size=(len(coords), 3))
    symbols = ['C'] * len(coords)
    atoms = Atoms(symbols, positions=coords)
    cell = np.array([[spacing*nx, 0.0, 0.0], [0.0, spacing*ny, 0.0], [0.0, 0.0, 10+spacing*nz]])
    atoms.set_cell(cell)
    return atoms

def create_dimer_system(separation=3.35):
    """Create a simple dimer system for testing force calculations."""
    # Create two carbon atoms at specified separation
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, separation]
    ])
    
    # Create a large cell to avoid periodic interactions
    cell = np.array([
        [20.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        [0.0, 0.0, 20.0]
    ])
    
    atoms = Atoms('CC', positions=positions, cell=cell, pbc=True)
    
    # Set mol-id to distinguish layers
    atoms.set_array('mol-id', np.array([1, 2]))
    
    return atoms

def test_finite_differences_vs_hellman_feynman():
    """Test finite differences vs hellman-feynman forces."""

    def V_ppσ(descriptors):
        """Bond integral from displacement descriptor"""
        # descriptors from get_disp is displacement [npairs, 3]
        # Compute distance from displacement
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return -2.7 * torch.exp(-(dist - 1.42) / 0.319)

    def V_ppπ(descriptors):
        """Bond integral from displacement descriptor"""
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return 0.48 * torch.exp(-(dist - 1.42) / 0.319)

    # Create hopping function
    bond_funcs = {'V_ppσ': V_ppσ, 'V_ppπ': V_ppπ}
    hopping_func = create_hopping_function(bond_funcs, basis='pz')
    model_dict = {"TB":{"hopping form":hopping_func,"onsite form":None,"descriptor form":get_disp},
                                "solver":"diagonalization",
                                "cutoff":5.0,"basis":"pz"}

    

    #test dimer system first
    print("--------------------------------")
    print("Testing dimer system...kmesh=(1,1,1)")
    print("--------------------------------")
    calc_hf = DD_TETB_model(model_dict=model_dict,kmesh=(1,1,1)) 
    atoms = create_dimer_system()
    atoms.calc = calc_hf
    energy, forces = calc_hf.get_total_energy(atoms)
    print(f"Total energy: {energy:.6f} eV")
    print(f"Hellman-Feynman forces: {np.round(forces,decimals=5)}")
    fd_forces = finite_difference_forces(atoms,calc_hf)
    print(f"Finite differences forces: {np.round(fd_forces,decimals=5)}")
    

    #test dimer system first
    print("--------------------------------")
    print("Testing dimer system...kmesh=(2,2,1)")
    print("--------------------------------")
    calc_hf = DD_TETB_model(model_dict=model_dict,kmesh=(2,2,1)) 
    atoms = create_dimer_system()
    atoms.calc = calc_hf
    energy, forces = calc_hf.get_total_energy(atoms)
    print(f"Total energy: {energy:.6f} eV")
    print(f"Hellman-Feynman forces: {np.round(forces,decimals=5)}")
    fd_forces = finite_difference_forces(atoms,calc_hf)
    print(f"Finite differences forces: {np.round(fd_forces,decimals=5)}")


    print("--------------------------------")
    print("Testing cubic mesh...kmesh=(1,1,1)")
    print("--------------------------------")
    calc_hf = DD_TETB_model(model_dict=model_dict,kmesh=(1,1,1)) 
    N = 3
    atoms = generate_cubic_mesh(N,N,N)
    atoms.calc = calc_hf
    energy, forces = calc_hf.get_total_energy(atoms)
    print(f"Total energy: {energy:.6f} eV")

    N = len(atoms)
    fd_forces = finite_difference_forces(atoms,calc_hf)

    print(f"Hellman-Feynman forces: {np.round(forces,decimals=5)}")
    print(f"Finite differences forces: {np.round(fd_forces,decimals=5)}")

    print("--------------------------------")
    print("Testing cubic mesh...kmesh=(2,2,1)")
    print("--------------------------------")
    calc_hf = DD_TETB_model(model_dict=model_dict,kmesh=(2,2,1)) 
    N = 3
    atoms = generate_cubic_mesh(N,N,N)
    atoms.calc = calc_hf
    energy, forces = calc_hf.get_total_energy(atoms)
    print(f"Total energy: {energy:.6f} eV")

    N = len(atoms)
    fd_forces = finite_difference_forces(atoms,calc_hf)

    print(f"Hellman-Feynman forces: {np.round(forces,decimals=5)}")
    print(f"Finite differences forces: {np.round(fd_forces,decimals=5)}")

def finite_difference_forces(atoms, calc, delta=1e-4):
    """Compute finite difference forces numerically."""
    positions = atoms.get_positions().copy()
    N = len(positions)
    F_fd = np.zeros_like(positions)
    
    for a in range(N):
        for k in range(3):
            disp = np.zeros_like(positions)
            disp[a, k] = delta

            # E(R + δ)
            atoms.set_positions(positions + disp)
            Ep = calc.get_total_energy(atoms)[0]

            # E(R - δ)
            atoms.set_positions(positions - disp)
            Em = calc.get_total_energy(atoms)[0]

            F_fd[a, k] = -(Ep - Em) / (2 * delta)

    atoms.set_positions(positions)  # restore
    return F_fd

def test_band_structure_tblg():
    """Test band structure calculation."""

    sep = 3.35
    a = 2.46
    n=5
    theta=9.4
    p_found, q_found, theta_comp = fg.twist.find_p_q(theta,a_tol=2e-1)
    atoms=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                    p=p_found,q=q_found,lat_con=a,sym=["C","C"],
                                    mass=[12.01,12.01],sep=sep,h_vac=20)
    
    print(f"System info:")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Cell shape: {atoms.cell.array}")
    print(f"  Twist angle: {theta_comp:.2f}°")
    
    # Create calculator with Hellman-Feynman method
    def V_ppσ(descriptors):
        """Bond integral from displacement descriptor"""
        # descriptors from get_disp is displacement [npairs, 3]
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return -2.7 * torch.exp(-(dist - 1.42) / 0.319)

    def V_ppπ(descriptors):
        """Bond integral from displacement descriptor"""
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return 0.48 * torch.exp(-(dist - 1.42) / 0.319)

    # Create hopping function
    bond_funcs = {'V_ppσ': V_ppσ, 'V_ppπ': V_ppπ}
    hopping_func = create_hopping_function(bond_funcs, basis='pz')
    model_dict = {"TB":{"hopping form":hopping_func,"onsite form":None,"descriptor form":get_disp},
                                "solver":"diagonalization",
                                "cutoff":5.0,"basis":"pz"}
    calc = DD_TETB_model(model_dict=model_dict) 
    
    # Test energy calculation first
    
    # Define k-path in fractional coordinates (reciprocal lattice units)
    # For hexagonal graphene, the high-symmetry points are:
    Gamma = [0.0, 0.0, 0.0]
    K = [1/3, 2/3, 0.0]  # Fixed: was [1/3,2/3,0] which is incorrect for hexagonal
    M = [0.5, 0.0, 0.0]
    Kprime = [2/3, 1/3, 0.0]  # Fixed: was [2/3,1/3,0] which is incorrect for hexagonal
    
    sym_pts = [K, Gamma, M, Kprime]
    nk = 100  # Increased for better resolution
    
    print(f"  K-points: {sym_pts}")
    
    # Generate k-path
    (kvec, k_dist, k_node) = k_path(torch.tensor(sym_pts, dtype=torch.float32), nk)
    kvec = kvec.numpy()
    k_dist = k_dist.numpy()
    k_node = k_node.numpy()
    
    print(f"  Generated {len(kvec)} k-points")
    print(f"  K-path shape: {kvec.shape}")
    
    # Calculate band structure
    try:
        evals = calc.get_band_structure(atoms, kvec)
        print(f"  Band structure shape: {evals.shape}")
    except Exception as e:
        print(f"  Error in band structure calculation: {e}")
        return
    
    # Plot band structure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate Fermi energy (average of middle two bands)
    nbands = evals.shape[0]
    if nbands % 2 == 0:
        efermi = (evals[nbands//2-1, 0] + evals[nbands//2, 0]) / 2
    else:
        efermi = evals[nbands//2, 0]
    
    print(f"  Number of bands: {nbands}")
    print(f"  Fermi energy: {efermi:.6f} eV")
    
    # Plot bands
    for n in range(nbands):
        ax.plot(k_dist, evals[n, :] - efermi, 'b-', linewidth=0.8, alpha=0.7)
    
    # Add high-symmetry point markers
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    # Set labels and formatting
    ax.set_xlim(k_node[0], k_node[-1])
    ax.set_ylim(-2.0, 2.0)
    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_xlabel('Path in k-space')
    ax.set_title(f'{theta:.1f}° twisted bilayer graphene')
    
    # Add vertical lines at high-symmetry points
    for i, (x, label) in enumerate(zip(k_node, ['K', r'$\Gamma$', 'M', r"$K'$"])):
        ax.axvline(x=x, color='k', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.text(x, ax.get_ylim()[1]*0.9, label, ha='center', va='bottom', fontsize=12)
    
    # Set x-ticks
    ax.set_xticks(k_node)
    ax.set_xticklabels(['K', r'$\Gamma$', 'M', r"$K'$"])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/popov_theta_{theta}_graphene_band_structure.png", 
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()
    
    # Print some diagnostic information
    print(f"  Band gap at Gamma: {(evals[nbands//2, len(kvec)//2] - evals[(nbands-1)//2,len(kvec)//2]):.6f} eV")
    print(f"  Band gap at K: {(evals[nbands//2, 0] - evals[(nbands-1)//2,0]):.6f} eV")

def test_band_structure_sp3():
    """Test band structure calculation."""

    
    atoms=graphene(vacuum=10)
    
    print(f"System info:")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Cell shape: {atoms.cell.array}")
    
    # Create calculator with Hellman-Feynman method
    def V_ppσ(descriptors):
        """Bond integral from displacement descriptor"""
        # descriptors from get_disp is displacement [npairs, 3]
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return -2.7 * torch.exp(-(dist - 1.42) / 0.319)

    def V_ppπ(descriptors):
        """Bond integral from displacement descriptor"""
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return 0.48 * torch.exp(-(dist - 1.42) / 0.319)

    def V_ssσ(descriptors):
        """Bond integral from displacement descriptor"""
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return -3.7 * torch.exp(-(dist - 1.42) / 0.319)

    def V_spσ(descriptors):
        """Bond integral from displacement descriptor"""
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return 0.82 * torch.exp(-(dist - 1.42) / 0.319)

    # Create hopping function
    bond_funcs = {'V_ppσ': V_ppσ, 'V_ppπ': V_ppπ, 'V_ssσ': V_ssσ, 'V_spσ': V_spσ}
    hopping_func = create_hopping_function(bond_funcs, basis='sp')
    model_dict = {"TB":{"hopping form":hopping_func,"onsite form":None,"descriptor form":get_disp},
                                "solver":"diagonalization",
                                "cutoff":5.0,"basis":"sp"}
    calc = DD_TETB_model(model_dict=model_dict) 
    
    # Test energy calculation first
    
    # Define k-path in fractional coordinates (reciprocal lattice units)
    # For hexagonal graphene, the high-symmetry points are:
    Gamma = [0.0, 0.0, 0.0]
    K = [1/3, 1/3, 0.0]  # Fixed: was [1/3,2/3,0] which is incorrect for hexagonal
    M = [0.5, 0.0, 0.0]
    Kprime = [2/3, 2/3, 0.0]  # Fixed: was [2/3,1/3,0] which is incorrect for hexagonal
    
    sym_pts = [K, Gamma, M, Kprime]
    nk = 100  # Increased for better resolution
    
    print(f"  K-points: {sym_pts}")
    
    # Generate k-path
    (kvec, k_dist, k_node) = k_path(torch.tensor(sym_pts, dtype=torch.float32), nk)
    kvec = kvec.numpy()
    k_dist = k_dist.numpy()
    k_node = k_node.numpy()
    
    print(f"  Generated {len(kvec)} k-points")
    print(f"  K-path shape: {kvec.shape}")
    
    # Calculate band structure
    try:
        evals = calc.get_band_structure(atoms, kvec)
        print(f"  Band structure shape: {evals.shape}")
    except Exception as e:
        print(f"  Error in band structure calculation: {e}")
        return
    
    # Plot band structure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate Fermi energy (average of middle two bands)
    nbands = evals.shape[0]
    if nbands % 2 == 0:
        efermi = (evals[nbands//2-1, 0] + evals[nbands//2, 0]) / 2
    else:
        efermi = evals[nbands//2, 0]
    
    print(f"  Number of bands: {nbands}")
    print(f"  Fermi energy: {efermi:.6f} eV")
    
    # Plot bands
    for n in range(nbands):
        ax.plot(k_dist, evals[n, :] - efermi, 'b-', linewidth=0.8, alpha=0.7)
    
    # Add high-symmetry point markers
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    # Set labels and formatting
    ax.set_xlim(k_node[0], k_node[-1])
    #ax.set_ylim(-2.0, 2.0)
    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_xlabel('Path in k-space')
    ax.set_title(f'sp3 hybridized graphene')
    
    # Add vertical lines at high-symmetry points
    for i, (x, label) in enumerate(zip(k_node, ['K', r'$\Gamma$', 'M', r"$K'$"])):
        ax.axvline(x=x, color='k', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.text(x, ax.get_ylim()[1]*0.9, label, ha='center', va='bottom', fontsize=12)
    
    # Set x-ticks
    ax.set_xticks(k_node)
    ax.set_xticklabels(['K', r'$\Gamma$', 'M', r"$K'$"])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/graphene_band_structure_sp3.png", 
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()
    
    # Print some diagnostic information
    print(f"  Band gap at Gamma: {(evals[nbands//2, len(kvec)//2] - evals[(nbands-1)//2,len(kvec)//2]):.6f} eV")
    print(f"  Band gap at K: {(evals[nbands//2, 0] - evals[(nbands-1)//2,0]):.6f} eV")

def test_band_structure():
    """Test band structure calculation."""

    
    atoms=graphene(vacuum=10)
    
    print(f"System info:")
    print(f"  Number of atoms: {len(atoms)}")
    print(f"  Cell shape: {atoms.cell.array}")
    
    # Create calculator with Hellman-Feynman method
    def V_ppσ(descriptors):
        """Bond integral from displacement descriptor"""
        # descriptors from get_disp is displacement [npairs, 3]
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return -2.7 * torch.exp(-(dist - 1.42) / 0.319)

    def V_ppπ(descriptors):
        """Bond integral from displacement descriptor"""
        dist = torch.norm(descriptors, dim=1)  # [npairs]
        return 0.48 * torch.exp(-(dist - 1.42) / 0.319)

    # Create hopping function
    bond_funcs = {'V_ppσ': V_ppσ, 'V_ppπ': V_ppπ}
    hopping_func = create_hopping_function(bond_funcs, basis='pz')
    model_dict = {"TB":{"hopping form":hopping_func,"onsite form":None,"descriptor form":get_disp},
                                "solver":"diagonalization",
                                "cutoff":5.0,"basis":"pz"}
    calc = DD_TETB_model(model_dict=model_dict) 
    
    # Test energy calculation first
    
    # Define k-path in fractional coordinates (reciprocal lattice units)
    # For hexagonal graphene, the high-symmetry points are:
    Gamma = [0.0, 0.0, 0.0]
    K = [1/3, 1/3, 0.0]  # Fixed: was [1/3,2/3,0] which is incorrect for hexagonal
    M = [0.5, 0.0, 0.0]
    Kprime = [2/3, 2/3, 0.0]  # Fixed: was [2/3,1/3,0] which is incorrect for hexagonal
    
    sym_pts = [K, Gamma, M, Kprime]
    nk = 100  # Increased for better resolution
    
    print(f"  K-points: {sym_pts}")
    
    # Generate k-path
    (kvec, k_dist, k_node) = k_path(torch.tensor(sym_pts, dtype=torch.float32), nk)
    kvec = kvec.numpy()
    k_dist = k_dist.numpy()
    k_node = k_node.numpy()
    
    print(f"  Generated {len(kvec)} k-points")
    print(f"  K-path shape: {kvec.shape}")
    
    # Calculate band structure
    try:
        evals = calc.get_band_structure(atoms, kvec)
        print(f"  Band structure shape: {evals.shape}")
    except Exception as e:
        print(f"  Error in band structure calculation: {e}")
        return
    
    # Plot band structure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate Fermi energy (average of middle two bands)
    nbands = evals.shape[0]
    if nbands % 2 == 0:
        efermi = (evals[nbands//2-1, 0] + evals[nbands//2, 0]) / 2
    else:
        efermi = evals[nbands//2, 0]
    
    print(f"  Number of bands: {nbands}")
    print(f"  Fermi energy: {efermi:.6f} eV")
    
    # Plot bands
    for n in range(nbands):
        ax.plot(k_dist, evals[n, :] - efermi, 'b-', linewidth=0.8, alpha=0.7)
    
    # Add high-symmetry point markers
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    # Set labels and formatting
    ax.set_xlim(k_node[0], k_node[-1])
    #ax.set_ylim(-2.0, 2.0)
    ax.set_ylabel(r'$E - E_F$ (eV)')
    ax.set_xlabel('Path in k-space')
    ax.set_title(f'sp3 hybridized graphene')
    
    # Add vertical lines at high-symmetry points
    for i, (x, label) in enumerate(zip(k_node, ['K', r'$\Gamma$', 'M', r"$K'$"])):
        ax.axvline(x=x, color='k', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.text(x, ax.get_ylim()[1]*0.9, label, ha='center', va='bottom', fontsize=12)
    
    # Set x-ticks
    ax.set_xticks(k_node)
    ax.set_xticklabels(['K', r'$\Gamma$', 'M', r"$K'$"])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/graphene_band_structure.png", 
                bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()
    
    # Print some diagnostic information
    print(f"  Band gap at Gamma: {(evals[nbands//2, len(kvec)//2] - evals[(nbands-1)//2,len(kvec)//2]):.6f} eV")
    print(f"  Band gap at K: {(evals[nbands//2, 0] - evals[(nbands-1)//2,0]):.6f} eV")

if __name__ == "__main__":
    test_finite_differences_vs_hellman_feynman()
    test_band_structure()
    test_band_structure_sp3()
    test_band_structure_tblg()