import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import GDELTLite
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import List

class FGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_scales: int):
        """
        Fractal Graph Neural Network (FGNN) module.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            out_channels (int): Number of output channels.
            num_scales (int): Number of scales in the FGNN.
        """
        super(FGNN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels)] +
                                   [GCNConv(hidden_channels, hidden_channels) for _ in range(num_scales - 2)] +
                                   [GCNConv(hidden_channels, hidden_channels)])
        self.final_conv = GCNConv(hidden_channels, out_channels)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FGNN.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Output node features.
        """
        scale_features = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = nn.functional.relu(x)
            scale_features.append(x)
        
        scale_features = torch.stack(scale_features, dim=-1)
        scale_weights = nn.functional.softmax(self.scale_weights, dim=0)
        x = torch.sum(scale_features * scale_weights, dim=-1)
        x = self.final_conv(x, edge_index)
        return x

def train(model: FGNN, optimizer: optim.Optimizer, loader: DataLoader, epochs: int):
    """
    Training loop for the FGNN model.

    Args:
        model (FGNN): FGNN model to train.
        optimizer (optim.Optimizer): Optimizer for training.
        loader (DataLoader): DataLoader for training data.
        epochs (int): Number of training epochs.
    """
    model.train()
    for _ in tqdm(range(epochs)):
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x.float(), data.edge_index)  # Convert node features to float
            # Perform any desired training task or loss computation here
            # For example, you can use the output embeddings for downstream tasks like link prediction or node classification

def evaluate(model: FGNN, loader: DataLoader) -> float:
    """
    Evaluate the FGNN model on test data.

    Args:
        model (FGNN): FGNN model to evaluate.
        loader (DataLoader): DataLoader for test data.
    
    Returns:
        float: Mean squared erro (MSE) on test data.
    """
    model.eval()
    mse_sum = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x.float(), data.edge_index)
            mse_sum += nn.MSELoss()(out, data.y.float()).item()
        return mse_sum / len(loader)

def main():
    # Load the GDELTLite dataset
    dataset = GDELTLite(root='./data/GDELTLite')
    train_dataset = dataset[:100]
    test_dataset = dataset[100:200]

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model hyperparameters
    in_channels = dataset.num_node_features
    hidden_channels = 64
    out_channels = 1
    num_scales = 3
    epochs = 50

    # Create the FGNN model
    model = FGNN(in_channels, hidden_channels, out_channels, num_scales)

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    train(model, optimizer, train_loader, epochs)

    # Evaluate the model
    test_mse = evaluate(model, test_loader)
    print(f"Test MSE: {test_mse:.4f}")

    # Baseline models
    baseline_models = [
        ("GCN", GCNConv(in_channels, out_channels)),
        # ADd more baseline models here
    ]

    # Evaluate baseline models
    for name, baseline_model in baseline_models:
        baseline_optimizer = optim.Adam(baseline_model.parameters(), lr=0.01)
        train(baseline_model, baseline_optimizer, train_loader, epochs)
        baseline_mse = evaluate(baseline_model, test_loader)
        print(f"{name} Test MSE: {baseline_mse:.4f}")

if __name__ == "__main__":
    main()