import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from typing import Tuple, Optional
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

def to_dense_adj(edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None,
                 edge_attr: Optional[torch.Tensor] = None, max_num_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None) -> torch.Tensor:
    if batch is None:
        max_index = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        batch = edge_index.new_zeros(max_index)

    if batch_size is None:
        batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

    one = batch.new_ones(batch.size(0))
    num_nodes = torch.zeros(batch_size, dtype=torch.long, device=batch.device)
    num_nodes.scatter_add_(0, batch, one)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    size = [batch_size, max_num_nodes, max_num_nodes]

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    size += list(edge_attr.size())[1:]

    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)
    flattened_size = batch_size * max_num_nodes * max_num_nodes
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj.view(-1).scatter_add_(0, idx, edge_attr)

    return adj

class TransformerLayer(nn.Module):
    """
    Transformer layer that performs self-attention on the input tensor.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1) -> None:
        super(TransformerLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(in_channels, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, in_channels)
        )
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.self_attention(x, x, x)[0]
        x = self.norm1(attended + x)
        fed_forward = self.feed_forward(x)
        x = self.norm2(fed_forward + x)
        return self.dropout(x)

class SpatioTemporalFGNN(nn.Module):
    """
    Spatiotemporal Fractal Graph Neural Network (FGNN) model.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_scales: int, seq_length: int, dropout: float = 0.1) -> None:
        super(SpatioTemporalFGNN, self).__init__()
        self.seq_length = seq_length
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels, kernel_size=1)] +
                                   [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1) for _ in range(num_scales - 2)] +
                                   [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)])
        self.transformer = TransformerLayer(hidden_channels, hidden_channels, heads=4, dropout=dropout)
        self.linear = nn.Linear(hidden_channels * seq_length, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, num_nodes, _ = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_features, seq_length, num_nodes)
        
        for conv in self.convs:
            x = conv(x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
        
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch_size, seq_length, num_nodes, hidden_channels)

        dense_adj = to_dense_adj(edge_index, batch, max_num_nodes=num_nodes)
        x = self.transformer(x)
        
        x = x.view(batch_size, num_nodes, -1)
        x = self.linear(x)
        return x

def train(model: SpatioTemporalFGNN, optimizer: optim.Optimizer, loader: DataLoader, epochs: int, device: torch.device) -> None:
    """
    Train the FGNN model.
    """
    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = nn.MSELoss()(out, data.y)
            loss.backward()
            optimizer.step()

def evaluate(model: SpatioTemporalFGNN, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate the FGNN model.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
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
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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