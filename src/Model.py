import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from TB_Utils import *
from Descriptors import *
from NeighborList import *
from Solver import *
from TB_Utils import expand_atom_to_orbital_indices

# TODO:
# - validate HF forces (they seem mostly right rn)
# - validate Ewald energies and forces
# - write module to load data from unstructure current form (ase.io.read(,extyxy) -> atoms list, energies, hoppings etc.)
# - finish writing fit_MLP.py, make sure it works for hoppings and residual potential

class DD_TETB_model(Calculator):
    implemented_properties = ['energy','forces','potential_energy']
    ################################################################

    # ASE specific functions

    ################################################################
    def __init__(self,model_dict=None,kmesh=(1,1,1),**kwargs):
        Calculator.__init__(self,**kwargs)  
        if model_dict is None:
            self.model_dict = {"TB":{"hopping form":None,
                                    "onsite form":None,"descriptor form":None},
                                "Residual":{"residual form":None, "descriptor form":None}
                                "solver":"diagonalization",
                                "basis":"pz",
                                "cutoff":5.0}
        self.model_dict = model_dict
        self.kmesh = kmesh
        self.num_orbitals_per_atom = {"s":1,"pz":1,"p":3, "sp":4, "d":5, "spd":9}[self.model_dict["basis"]]

    def calculate(self, atoms, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        total_energy, forces = self.get_total_energy(atoms)
        self.results['forces'] = forces
        self.results['potential_energy'] = total_energy
        self.results['energy'] = total_energy

    def get_total_energy(self,atoms,energy_contributions = False):
        
        positions = torch.tensor(atoms.positions, dtype=torch.float32)
        cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32)
        disp, atom_i, atom_j, atom_di, atom_dj = get_neighbor_list(positions, cell, self.model_dict["cutoff"])
        #atom_i, atom_j are self inclusive neighbor lists
        atom_centered_descriptors = self.model_dict["descriptor form"](atoms) 
        atom_centered_descriptors.requires_grad_(True)

        tb_energy, tb_forces, mulliken_charges = self.get_tb_energy(positions, cell, atom_centered_descriptors,
                                                                     disp, atom_i, atom_j, atom_di, atom_dj)
        mulliken_charges = mulliken_charges.detach()
        residual_energy = self.get_residual_energy(positions, cell, atom_centered_descriptors, mulliken_charges,
                                                    disp, atom_i, atom_j, atom_di, atom_dj)
        residual_forces = torch.autograd.grad(
                                            residual_energy,
                                            positions,
                                            retain_graph=True,
                                            create_graph=False
                                        )[0]
        Ewald_energy, Ewald_forces = 0,0 #get_Ewald_energy(positions, cell, mulliken_charges)
        total_energy = tb_energy + residual_energy + Ewald_energy
        forces = tb_forces + residual_forces + Ewald_forces
        if energy_contributions:
            return (total_energy, forces), (tb_energy, tb_forces), (residual_energy, residual_forces), (Ewald_energy, Ewald_forces)
        else:
            return total_energy, forces

    def run(self, atoms):
        """Run calculation."""
        self.calculate(atoms)

    ################################################################

    # DD-TETB model functions

    ################################################################
    #@torch.jit.script
    def get_Hamiltonian(self, atom_centered_descriptors, positions, cell, neighbor_list, cell_offsets):
        #I think the best way to go about this is to have a neighbor list that is 
        # just a ragged array where nl[i] = [j1, j2, j3, ...] describes the neighbors of atom i
        # then there will be a for loop over atoms where we find 3 center integrals for nl[i]
        # this function will have to be jit compiled for speed I think
        Ham = torch.zeros((self.norbs, self.norbs), dtype=torch.complex64)
        hop_i, hop_j = expand_atom_to_orbital_indices(atom_i, atom_j, self.num_orbitals_per_atom)
        model_output = self.model_dict["TB"]["model"](atom_centered_descriptors)
        V_coeffs, V_spread, orbital_spread = model_output
        for i in range(len(positions)):
            neighbors = neighbor_list[i]
            Ra = positions[i]
            Rb = positions[neighbors] + cell_offsets[i] @ cell
            Rc = positions[neighbors] + cell_offsets[i] @ cell

            alpha = orbital_spread[i]
            beta = orbital_spread[neighbors]
            gamma = V_spread[neighbors]
            V0 = V_coeffs[neighbors]
            w = alpha + beta + gamma

            i0 = V0 * (np.pi / w)**1.5 * np.exp(-phi)
            Da = Q - Ra
            Db = Q - Rb
        
        
        # 6. Math: Universal Gaussian Integrals
        r_ab2 = np.sum((Ra - Rb)**2, axis=1)
        r_bv2 = np.sum((Rb - Rv)**2, axis=1)
        r_av2 = np.sum((Ra - Rv)**2, axis=1)
        
        phi = (alpha*beta*r_ab2 + beta*gamma*r_bv2 + alpha*gamma*r_av2) / w
        i0 = V0 * (np.pi / w)**1.5 * np.exp(-phi)
        
        # Weighted center Q and Displacements D
        Q = (alpha[:, None]*Ra + beta[:, None]*Rb + gamma*Rv) / w[:, None]
        Da = Q - Ra
        Db = Q - Rb
        
        # 7. Scatter into Hamiltonian
        # Note: We scatter based on local indices (a_tri, b_tri) 
        # The PBC physics is already inside the i0 and D terms.
        
        # s-s
        Ham.index_put_((hop_i, hop_j), i0 / (4 * np.pi), accumulate=True)
        
        # p-s and s-p
        n_p = np.sqrt(3) / (4 * np.pi)
        for i in range(3):
            Ham.index_put_((hop_i + 1 + i, hop_j), i0 * n_p * Da[:, i], accumulate=True)
            Ham.index_put_((hop_i, hop_j + 1 + i), i0 * n_p * Db[:, i], accumulate=True)
            
        # p-p
        n_pp = 3.0 / (4 * np.pi)
        for i in range(3):
            for k in range(3):
                delta = 1.0 if i == k else 0.0
                pp_vals = i0 * n_pp * (Da[:, i] * Db[:, k] + (delta / (2 * w)))
                Ham.index_put_((hop_i + 1 + i, hop_j + 1 + k), pp_vals, accumulate=True)
        return Ham, hop_i, hop_j

    def get_tb_energy(self, positions, cell, atom_centered_descriptors,
                      disp, atom_i, atom_j, atom_di, atom_dj):
        # Enable gradients for positions to compute hopping derivatives
        positions.requires_grad_(True)
        
        kpoints_reduced = k_uniform_mesh(self.kmesh).to(dtype=torch.float32)
        recip_cell = get_recip_cell(cell)
        kpoints = kpoints_reduced @ recip_cell
        nkp = len(kpoints)
        natoms = len(positions)
        tb_energy = 0
        forces = torch.zeros((natoms, 3), dtype=torch.float32)
        self.norbs = natoms * self.num_orbitals_per_atom
        mulliken_charges = torch.zeros(self.norbs, dtype=torch.float32)


        Ham_Gamma, hop_i, hop_j = self.get_Hamiltonian(atom_centered_descriptors, positions, cell, atom_i, atom_j, atom_di, atom_dj)
        # Loop over k-points to compute energy and forces
        for k_idx in range(nkp):  # TODO: parallelize this loop
            # Initialize Hamiltonian
            ham = Ham_Gamma.clone()
            
            # Add hopping terms with phase
            phase = torch.exp(1.0j * torch.sum(kpoints[k_idx, :].unsqueeze(0) * self.disp, dim=1))
            
            # Expand phase to match expanded hoppings
            # Each pair has n_orbitals^2 hoppings, so repeat phase accordingly
            phase_expanded = phase.repeat_interleave(self.num_orbitals_per_atom ** 2)
            
            ham.index_put_((hop_i, hop_j), phase_expanded, accumulate=True)
            ham.index_put_((hop_j, hop_i), phase_expanded.conj(), accumulate=True)

            # Solve Hamiltonian
            density_matrix = Solve_Hamiltonian(ham, self.model_dict["solver"], return_eigvals=False)
            tb_energy += torch.trace(density_matrix@ham).real/nkp
            forces += torch.autograd.grad(
                tb_energy,
                positions,
                retain_graph=True,
                create_graph=False
            )[0]/nkp
            mulliken_charges += torch.diag(density_matrix).real/nkp #contract this from [norbs] to [natoms]
        return tb_energy, forces.detach(), mulliken_charges.detach()

    def get_band_structure(self, atoms, reduced_kpoints):
        positions = torch.tensor(atoms.positions, dtype=torch.float32)
        cell = torch.tensor(np.array(atoms.cell), dtype=torch.float32)
        
        # Get neighbor list
        disp, atom_i, atom_j, atom_di, atom_dj = get_neighbor_list(positions, cell, self.model_dict["cutoff"])
        
        # Get descriptors and hoppings
        descriptors = self.model_dict["TB"]["descriptor form"](positions, cell, atom_i, atom_j, atom_di, atom_dj)
        hoppings = self.model_dict["TB"]["hopping form"](descriptors, disp)
        
        # Set norbs
        norbs = len(positions) * self.num_orbitals_per_atom
        
        # Onsite energies
        if self.model_dict["TB"]["onsite form"] is not None:
            onsite_energies = self.model_dict["TB"]["onsite form"](positions, cell)
        else:
            onsite_energies = torch.zeros(norbs, dtype=torch.float32)
        
        # Expand atom indices to orbital indices
        hop_i, hop_j, hoppings = expand_atom_to_orbital_indices(
            atom_i, atom_j, hoppings, self.num_orbitals_per_atom
        )
        
        # Convert reduced k-points to Cartesian
        reduced_kpoints = torch.tensor(reduced_kpoints, dtype=torch.float32)
        recip_cell = get_recip_cell(cell)
        kpoints = reduced_kpoints @ recip_cell
        nkp = len(kpoints)
        
        # Initialize eigenvalue array
        eigvals_k = torch.zeros((norbs, nkp), dtype=torch.float32)

        for k_idx in range(nkp):
            # Initialize Hamiltonian
            ham = torch.diag(onsite_energies).to(dtype=torch.complex64)
            
            # Add hopping terms with phase
            phase = torch.exp(1.0j * torch.sum(kpoints[k_idx, :].unsqueeze(0) * disp, dim=1))
            
            # Expand phase to match expanded hoppings
            # Each pair has n_orbitals^2 hoppings, so repeat phase accordingly
            phase_expanded = phase.repeat_interleave(self.num_orbitals_per_atom ** 2)
            
            amp = hoppings.to(dtype=torch.complex64) * phase_expanded
            ham.index_put_((hop_i, hop_j), amp, accumulate=True)
            ham.index_put_((hop_j, hop_i), amp.conj(), accumulate=True)
            
            # Solve Hamiltonian
            density_matrix, eigvals = Solve_Hamiltonian(ham, self.model_dict["solver"], return_eigvals=True)
            eigvals_k[:, k_idx] = eigvals.real
            
        return eigvals_k.detach().cpu().numpy()