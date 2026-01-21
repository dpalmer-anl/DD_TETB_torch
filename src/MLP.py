import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    """
    General MLP-based function that predicts a value from a vector of descriptors.
    
    Input: descriptors (vector)
    Output: predicted value
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, descriptors):
        """
        Args:
            descriptors: descriptor values
        Returns:
            output value: 
        """
        if descriptors.dim() == 1:
            descriptors = descriptors.unsqueeze(1)  # [N_pairs, 1]
        return self.mlp(descriptors)  # [N_pairs, 2]

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader
    
    print("Generating training data...")
    
    # Generate displacement vectors
    rcut = 5.0
    X, Y, Z = np.meshgrid(
        np.linspace(0.5, rcut, 20),
        np.linspace(0.5, rcut, 20),
        np.linspace(0.5, rcut, 20)
    )
    displacements_np = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    # Filter to within cutoff
    displacements_np = displacements_np[np.linalg.norm(displacements_np, axis=1) < rcut]
    
    # Convert to torch
    displacements = torch.tensor(displacements_np, dtype=torch.float32)
    
    # Compute distances (input to MLP)
    distances = torch.norm(displacements, dim=1)  # [npairs]
    
    # Compute direction cosines (for SK transformation)
    n = displacements / distances.unsqueeze(1)  # [npairs, 3] normalized directions
    n_z = n[:, 2]  # z-component for pz-pz hopping
    
    # Compute target bond integrals (what we want MLP to learn)
    target_Vσ = V_ppσ(displacements)  # [npairs]
    target_Vπ = V_ppπ(displacements)  # [npairs]
    
    # Compute target hoppings using SK transformation: t = n_z^2 * Vπ + (1 - n_z^2) * Vσ
    target_hoppings = n_z**2 * target_Vπ + (1 - n_z**2) * target_Vσ  # [npairs]
    
    print(f"Generated {len(distances)} training samples")
    print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}] Å")
    print(f"Target Vσ range: [{target_Vσ.min():.4f}, {target_Vσ.max():.4f}] eV")
    print(f"Target Vπ range: [{target_Vπ.min():.4f}, {target_Vπ.max():.4f}] eV")
    print(f"Target hopping range: [{target_hoppings.min():.4f}, {target_hoppings.max():.4f}] eV")
    
    # Store n_z for each sample (needed for loss calculation)
    # Create dataset: (distance, n_z, target_hopping)
    dataset = TensorDataset(distances, n_z, target_hoppings)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Training parameters
    num_epochs = 1000
    learning_rate = 0.001
    
    # Initialize model (input_dim=1 for distance, output_dim=2 for [Vσ, Vπ])
    model = MLP(input_dim=1, hidden_dim=64, output_dim=2)
    criterion = nn.MSELoss()  # Mean squared error for hopping values
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining MLP with {sum(p.numel() for p in model.parameters())} parameters...")
    print("Model architecture: distance -> bond_integrals[Vσ, Vπ] -> hopping")
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for dist_batch, nz_batch, target_hopping_batch in train_loader:
            # Forward pass: predict bond integrals
            pred_bond_integrals = model(dist_batch)  # [batch, 2] -> [Vσ, Vπ]
            
            # Apply Slater-Koster transformation: t = n_z^2 * Vπ + (1 - n_z^2) * Vσ
            pred_hopping = (nz_batch**2 * pred_bond_integrals[:, 1] + 
                           (1 - nz_batch**2) * pred_bond_integrals[:, 0])
            
            # Compute loss on hopping values
            loss = criterion(pred_hopping, target_hopping_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    with torch.no_grad():
        pred_bond_integrals = model(distances)  # [npairs, 2] -> [Vσ, Vπ]
        pred_Vσ = pred_bond_integrals[:, 0]
        pred_Vπ = pred_bond_integrals[:, 1]
        
        # Apply SK transformation
        pred_hoppings = n_z**2 * pred_Vπ + (1 - n_z**2) * pred_Vσ
    
    # Calculate metrics
    hopping_mse = torch.mean((pred_hoppings - target_hoppings)**2)
    hopping_mae = torch.mean(torch.abs(pred_hoppings - target_hoppings))
    hopping_r2 = 1 - hopping_mse / torch.var(target_hoppings)
    
    Vσ_mae = torch.mean(torch.abs(pred_Vσ - target_Vσ))
    Vπ_mae = torch.mean(torch.abs(pred_Vπ - target_Vπ))
    
    print(f"\nFinal metrics:")
    print(f"  Hopping MSE: {hopping_mse:.6f}")
    print(f"  Hopping MAE: {hopping_mae:.6f}")
    print(f"  Hopping R²:  {hopping_r2:.6f}")
    print(f"  Vσ MAE:      {Vσ_mae:.6f}")
    print(f"  Vπ MAE:      {Vπ_mae:.6f}")
    
    # Plotting
    print("\nGenerating plots...")
    
    distances_np = distances.numpy()
    
    # Create comprehensive plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Bond integrals - Vσ
    plt.subplot(2, 3, 1)
    plt.scatter(distances_np, target_Vσ.numpy(), alpha=0.3, s=10, label='True Vσ', c='blue')
    plt.scatter(distances_np, pred_Vσ.numpy(), alpha=0.3, s=10, label='Pred Vσ', c='red')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Vσ (eV)')
    plt.title('Bond Integral Vσ vs Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Bond integrals - Vπ
    plt.subplot(2, 3, 2)
    plt.scatter(distances_np, target_Vπ.numpy(), alpha=0.3, s=10, label='True Vπ', c='blue')
    plt.scatter(distances_np, pred_Vπ.numpy(), alpha=0.3, s=10, label='Pred Vπ', c='red')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Vπ (eV)')
    plt.title('Bond Integral Vπ vs Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Hopping values
    plt.subplot(2, 3, 3)
    plt.scatter(distances_np, target_hoppings.numpy(), alpha=0.3, s=10, label='True', c='blue')
    plt.scatter(distances_np, pred_hoppings.numpy(), alpha=0.3, s=10, label='Predicted', c='red')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Hopping (eV)')
    plt.title('Hopping vs Distance (after SK transform)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Vσ error
    plt.subplot(2, 3, 4)
    error_Vσ = (pred_Vσ - target_Vσ).numpy()
    plt.scatter(distances_np, error_Vσ, alpha=0.3, s=10, c='green')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Vσ Error (eV)')
    plt.title(f'Vσ Prediction Error (MAE={Vσ_mae:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Vπ error
    plt.subplot(2, 3, 5)
    error_Vπ = (pred_Vπ - target_Vπ).numpy()
    plt.scatter(distances_np, error_Vπ, alpha=0.3, s=10, c='green')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Vπ Error (eV)')
    plt.title(f'Vπ Prediction Error (MAE={Vπ_mae:.4f})')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Hopping error
    plt.subplot(2, 3, 6)
    error_hopping = (pred_hoppings - target_hoppings).numpy()
    plt.scatter(distances_np, error_hopping, alpha=0.3, s=10, c='orange')
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Hopping Error (eV)')
    plt.title(f'Hopping Prediction Error (MAE={hopping_mae:.4f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/MLP_hopping_function.png", dpi=150)
    print("Saved plot to figures/MLP_hopping_function.png")
    
    # Save model
    torch.save(model.state_dict(), 'mlp_hopping_model.pth')
    print("Saved model to mlp_hopping_model.pth")