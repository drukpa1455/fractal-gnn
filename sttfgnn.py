import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from typing import Tuple
import numpy as np

class SpatioTemporalFGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_scales: int, seq_length: int, dropout: float = 0.1) -> None:
        super(SpatioTemporalFGNN, self).__init__()
        self.seq_length = seq_length
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels)] +
                                   [GCNConv(hidden_channels, hidden_channels) for _ in range(num_scales - 2)] +
                                   [GCNConv(hidden_channels, hidden_channels)])
        self.transformer = TransformerConv(hidden_channels, hidden_channels, heads=4, dropout=dropout)
        self.linear = nn.Linear(hidden_channels * seq_length, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, num_nodes, _ = x.shape
        x = x.view(batch_size * seq_length, num_nodes, -1)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = nn.functional.relu(x)
            x = self.dropout(x)
        
        x = x.view(batch_size, seq_length, num_nodes, -1)
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_nodes, seq_length, hidden_channels)
        x = self.transformer(x, edge_index)
        
        x = x.view(batch_size, num_nodes, -1)
        x = self.linear(x)
        return x

def train(model: SpatioTemporalFGNN, optimizer: optim.Optimizer, loader: DataLoader, epochs: int, device: torch.device) -> None:
    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = nn.MSELoss()(out, data.y)
            loss.backward()
            optimizer.step()

def evaluate(model: SpatioTemporalFGNN, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            y_true.append(data.y.view(-1))
            y_pred.append(out.view(-1))
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    mse = mean_squared_error(y_true.cpu().numpy(), y_pred.cpu().numpy())
    mae = mean_absolute_error(y_true.cpu().numpy(), y_pred.cpu().numpy())
    return mse, mae

def main() -> None:
    # Set the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the METR-LA dataset
    loader = METRLADatasetLoader()
    dataset = loader.get_dataset()

    # Split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.6, val_ratio=0.2)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Model hyperparameters
    in_channels = 2  # Traffic speed and timestamp
    hidden_channels = 64
    out_channels = 1  # Predicted traffic speed
    num_scales = 3
    seq_length = 12  # Number of historical timesteps
    dropout = 0.1
    epochs = 100
    lr = 0.001
    weight_decay = 1e-5

    # Create the spatiotemporal FGNN model
    model = SpatioTemporalFGNN(in_channels, hidden_channels, out_channels, num_scales, seq_length, dropout).to(device)

    # Create the optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model with early stopping
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(epochs):
        train(model, optimizer, train_loader, 1, device)
        val_mse, val_mae = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}, Validation MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model on the test set
    test_mse, test_mae = evaluate(model, test_loader, device)
    print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

if __name__ == "__main__":
    main()