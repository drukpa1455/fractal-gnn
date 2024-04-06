import os
import urllib.request
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, TransformerConv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from typing import Tuple

class METRLADataset(Dataset):
    def __init__(self, data_dir: str, num_timesteps_in: int = 12, num_timesteps_out: int = 12) -> None:
        self.data_dir = data_dir
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out
        self._download_data()
        self.data, self.indices = self._load_data()
        self.train_indices = self.indices['train']
        self.val_indices = self.indices['val']
        self.test_indices = self.indices['test']
        
    def _download_data(self) -> None:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(os.path.join(self.data_dir, 'data.npz')):
            urllib.request.urlretrieve("https://github.com/XDZhelheim/STAEformer/raw/main/data/METRLA/data.npz", os.path.join(self.data_dir, 'data.npz'))
        if not os.path.exists(os.path.join(self.data_dir, 'index.npz')):
            urllib.request.urlretrieve("https://github.com/XDZhelheim/STAEformer/raw/main/data/METRLA/index.npz", os.path.join(self.data_dir, 'index.npz'))
                
    def _load_data(self) -> Tuple[np.ndarray, dict]:
        data = np.load(os.path.join(self.data_dir, 'data.npz'))['data']
        indices = np.load(os.path.join(self.data_dir, 'index.npz'), allow_pickle=True)['index'].item()
        return data, indices
        
    def __len__(self) -> int:
        return len(self.train_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index = self.train_indices[idx]
        x = self.data[:, :, index:index+self.num_timesteps_in]
        y = self.data[:, :, index+self.num_timesteps_in:index+self.num_timesteps_in+self.num_timesteps_out]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    
    def get_val_indices(self) -> np.ndarray:
        return self.val_indices
    
    def get_test_indices(self) -> np.ndarray:
        return self.test_indices

class GraphAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timestep_max: int, time_filter: int, device: str):
        super().__init__()

        self.k = nn.Conv2d(in_channels=in_channels, out_channels=timestep_max, kernel_size=(1, time_filter), stride=1, device=device)
        self.q = nn.Conv2d(in_channels=in_channels, out_channels=timestep_max, kernel_size=(1, time_filter), stride=1, device=device)
        self.v = nn.Conv2d(in_channels=in_channels, out_channels=timestep_max, kernel_size=(1, time_filter), stride=1, device=device)

        self.fc_res = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, device=device)
        self.fc_out = nn.Conv2d(in_channels=(timestep_max - time_filter) + 1, out_channels=out_channels, kernel_size=(1, 1), stride=1, device=device)

        self.act = nn.Softmax(dim=-1)
        self.norm = nn.BatchNorm2d(out_channels, device=device)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None, adj_hat: torch.Tensor | None) -> torch.Tensor:
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        score = torch.einsum("BTNC, BTnC -> BTNn", k, q).contiguous()

        if adj is not None:
            score = score + adj.unsqueeze(1)
        if adj_hat is not None:
            score = score + adj_hat.unsqueeze(1)

        score = self.act(score)
        out = torch.einsum("BTnN, BTNC -> BCnT", score, v).contiguous()

        out = self.fc_out(out)

        res = self.fc_res(x)

        out = self.norm((out + res))

        out = self.dropout(out)

        return out

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, timestep_max: int, device: str):
        super().__init__()

        self.ga_2 = GraphAttention(in_channels=in_channels, out_channels=out_channels, timestep_max=timestep_max, time_filter=2, device=device)
        self.ga_3 = GraphAttention(in_channels=in_channels, out_channels=out_channels, timestep_max=timestep_max, time_filter=3, device=device)
        self.ga_6 = GraphAttention(in_channels=in_channels, out_channels=out_channels, timestep_max=timestep_max, time_filter=6, device=device)
        self.ga_7 = GraphAttention(in_channels=in_channels, out_channels=out_channels, timestep_max=timestep_max, time_filter=7, device=device)

        self.fc_res = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, device=device)
        self.fc_out = nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=(1, 1), stride=1, device=device)

        self.norm = nn.BatchNorm2d(out_channels, device=device)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None, adj_hat: torch.Tensor | None) -> torch.Tensor:
        res = self.fc_res(x)

        x = torch.cat([self.ga_2(x, adj, adj_hat), self.ga_3(x, adj, adj_hat), self.ga_6(x, adj, adj_hat), self.ga_7(x, adj, adj_hat)], dim=1)

        x = self.fc_out(x)

        x = self.norm((x + res))

        x = self.dropout(x)

        return x

class SpatioTemporalFGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_nodes: int, num_scales: int, seq_length: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels)] +
                                   [GCNConv(hidden_channels, hidden_channels) for _ in range(num_scales - 2)] +
                                   [GCNConv(hidden_channels, hidden_channels)])
        self.transformer = TransformerConv(hidden_channels, hidden_channels, heads=4, dropout=dropout)
        self.linear = nn.Linear(hidden_channels * seq_length, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.mhga = MultiHeadGraphAttention(in_channels=hidden_channels, out_channels=hidden_channels, timestep_max=seq_length, device=device)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.seq_length, -1)  # (batch_size, seq_length, num_nodes * num_features)
        x = x.permute(0, 2, 1)  # (batch_size, num_nodes * num_features, seq_length)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = nn.functional.relu(x)
            x = self.dropout(x)
        
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, num_nodes * hidden_channels)
        x = self.transformer(x, edge_index)
        
        x = self.mhga(x, adj=None, adj_hat=None)
        
        x = x.reshape(x.shape[0], -1)  # (batch_size, seq_length * hidden_channels)
        x = self.linear(x)
        return x

def train(model: SpatioTemporalFGNN, optimizer: torch.optim.Optimizer, loader: DataLoader, epochs: int, device: torch.device) -> None:
    model.train()
    progress_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    for _ in progress_bar:
        epoch_loss = 0.0
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features, edge_index=None)  # Assuming the edge_index is not used in this case
            loss = nn.MSELoss()(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": epoch_loss / len(loader)})

def evaluate(model: SpatioTemporalFGNN, data: np.ndarray, indices: np.ndarray, device: torch.device) -> Tuple[float, float]:
    model.eval()
    targets_all, outputs_all = [], []
    with torch.no_grad():
        for index in indices:
            features = torch.from_numpy(data[:, :, index:index+12]).float().to(device)
            targets = torch.from_numpy(data[:, :, index+12:index+24]).float().to(device)
            outputs = model(features, edge_index=None)  # Assuming the edge_index is not used in this case
            targets_all.append(targets.cpu().numpy())
            outputs_all.append(outputs.cpu().numpy())
    targets_all = np.concatenate(targets_all)
    outputs_all = np.concatenate(outputs_all)
    mse = mean_squared_error(targets_all, outputs_all)
    mae = mean_absolute_error(targets_all, outputs_all)
    return mse, mae

def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join(os.getcwd(), "data", "METRLA")
    num_timesteps_in, num_timesteps_out = 12, 12
    dataset = METRLADataset(data_dir, num_timesteps_in, num_timesteps_out)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    in_channels = dataset.data.shape[1]
    hidden_channels, out_channels = 64, num_timesteps_out
    num_nodes = dataset.data.shape[0]
    num_scales = 3
    seq_length = num_timesteps_in
    dropout, epochs, lr, weight_decay = 0.1, 100, 0.001, 1e-5
    model = SpatioTemporalFGNN(in_channels, hidden_channels, out_channels, num_nodes, num_scales, seq_length, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    patience, counter = 10, 0
    for epoch in range(epochs):
        train(model, optimizer, train_loader, 1, device)
        val_mse, _ = evaluate(model, dataset.data, dataset.get_val_indices(), device)
        print(f"Epoch {epoch+1}, Validation MSE: {val_mse:.4f}")
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(torch.load('best_model.pth'))
    test_mse, _ = evaluate(model, dataset.data, dataset.get_test_indices(), device)
    print(f"Test MSE: {test_mse:.4f}")

if __name__ == "__main__":
    main()