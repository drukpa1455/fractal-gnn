import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import GDELTLite
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from typing import Tuple

class FGNN(nn.Module):
    """
    Fractal Graph Neural Network (FGNN) module.

    Args:
        in_channels (int): Number of input channels (node features).
        hidden_channels (int): Number of hidden channels in the GCN layers.
        out_channels (int): Number of output channels (for the final prediction task).
        num_scales (int): Number of scales (GCN layers) in the FGNN architecture.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_scales: int) -> None:
        super(FGNN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels)] +
                                   [GCNConv(hidden_channels, hidden_channels) for _ in range(num_scales - 2)] +
                                   [GCNConv(hidden_channels, hidden_channels)])
        self.final_conv = GCNConv(hidden_channels, out_channels)
        self.scale_weights = nn.Parameter(torch.ones(num_scales))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FGNN model.

        Args:
            x (torch.Tensor): Input node features.
            edge_index (torch.Tensor): Edge indices representing the graph structure.

        Returns:
            torch.Tensor: Output node features after the FGNN layers.
        """
        scale_features = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = nn.functional.relu(x)
            scale_features.append(x)

        scale_features = torch.stack(scale_features, dim=1)  # (num_nodes, num_scales, hidden_channels)
        scale_weights = nn.functional.softmax(self.scale_weights, dim=0)
        x = torch.sum(scale_features * scale_weights.view(1, -1, 1), dim=1)  # (num_nodes, hidden_channels)
        x = self.final_conv(x, edge_index)
        return x

def train(model: FGNN, optimizer: optim.Optimizer, loader: DataLoader, epochs: int) -> None:
    """
    Training loop for the FGNN model.

    Args:
        model (FGNN): FGNN model to train.
        optimizer (optim.Optimizer): Optimizer for training.
        loader (DataLoader): DataLoader for training data.
        epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        for data in loader:
            optimizer.zero_grad()
            out = model(data.x.float(), data.edge_index)
            # Perform a self-supervised task or use an alternative loss function
            # Example: Reconstruct the input features
            loss = nn.MSELoss()(out, data.x.float())
            loss.backward()
            optimizer.step()

def evaluate(model: FGNN, loader: DataLoader) -> float:
    """
    Evaluate the FGNN model on a dataset.

    Args:
        model (FGNN): FGNN model to evaluate.
        loader (DataLoader): DataLoader for the evaluation dataset.

    Returns:
        float: Mean squared error (MSE) on the evaluation dataset.
    """
    model.eval()
    mse_sum = 0.0
    with torch.no_grad():
        for data in loader:
            out = model(data.x.float(), data.edge_index)
            mse_sum += nn.MSELoss()(out, data.x.float()).item()
    return mse_sum / len(loader)

def downstream_task(model: FGNN, loader: DataLoader) -> Tuple[float, float, float, float]:
    """
    Perform downstream task evaluation (e.g., link prediction) using the FGNN model.

    Args:
        model (FGNN): FGNN model to evaluate.
        loader (DataLoader): DataLoader for the evaluation dataset.

    Returns:
        tuple: Accuracy, Precision, Recall, F1-score for the downstream task.
    """
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x.float(), data.edge_index)
            # Perform link prediction based on the learned node embeddings
            # Example: Compute the dot product of node embeddings to predict links
            edge_scores = torch.sigmoid((out[data.edge_index[0]] * out[data.edge_index[1]]).sum(dim=1))
            predictions.extend(edge_scores.ge(0.5).tolist())
            true_labels.extend([1] * data.edge_index.size(1))  # Assume all existing edges are positive

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    return accuracy, precision, recall, f1

def main():
    # Load the GDELTLite dataset
    dataset = GDELTLite(root='./data/GDELTLite')

    # Split the dataset into train and test sets
    transform = RandomLinkSplit(is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(dataset.data)

    # Convert the data into lists
    train_data = [train_data]
    val_data = [val_data]
    test_data = [test_data]

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Model hyperparameters
    in_channels = dataset.num_node_features
    hidden_channels = 64
    out_channels = dataset.num_node_features  # Output size should match input size for reconstruction
    num_scales = 3
    epochs = 50
    lr = 0.01

    # Create the FGNN model
    model = FGNN(in_channels, hidden_channels, out_channels, num_scales)

    # Create the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    train(model, optimizer, train_loader, epochs)

    # Evaluate the model on test set (MSE)
    test_mse = evaluate(model, test_loader)
    print(f"\nTest MSE: {test_mse:.4f}")

    # Perform downstream task evaluation (link prediction) on test set
    accuracy, precision, recall, f1 = downstream_task(model, test_loader)
    print("\nDownstream Task Evaluation (Link Prediction):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

if __name__ == "__main__":
    main()