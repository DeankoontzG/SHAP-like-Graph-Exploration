import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
import os

class HybridLinkPredictor(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(node_in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        input_mlp = (hidden_channels * 2) + edge_in_channels
        self.mlp = nn.Sequential(
            nn.Linear(input_mlp, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1)
        )

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index, edge_attr):
        u, v = edge_label_index[0], edge_label_index[1]
        combined = torch.cat([z[u], z[v], edge_attr], dim=-1)
        return self.mlp(combined).view(-1)

    def train_step(self, data, optimizer, criterion):
        self.train()
        optimizer.zero_grad()
        z = self.encode(data.x, data.edge_index)
        out = self.decode(z, data.edge_label_index, data.edge_attr)
        loss = criterion(out, data.edge_label)
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test(self, data):
        self.eval()
        z = self.encode(data.x, data.edge_index)
        out = self.decode(z, data.edge_label_index, data.edge_attr)
        y_pred = torch.sigmoid(out).cpu().numpy()
        y_true = data.edge_label.cpu().numpy()
        return roc_auc_score(y_true, y_pred)

def run_gnn_training(data, epochs=1000, lr=0.005):
    # Sélection automatique du device (MPS pour Mac, CUDA pour GPU, sinon CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"🚀 Entraînement sur : {device}")
    
    data = data.to(device)
    
    model = HybridLinkPredictor(
        node_in_channels=data.x.size(1),
        edge_in_channels=data.edge_attr.size(1),
        hidden_channels=64
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = 0
    model_save_path = "outputs/models/hybrid_gnn_model.pth"

    for epoch in range(1, epochs + 1):
        loss = model.train_step(data, optimizer, criterion)
        
        if epoch % 10 == 0 or epoch == 1:
            current_auc = model.test(data)
            if current_auc > best_auc:
                best_auc = current_auc
                # On sauvegarde l'état du meilleur modèle
                torch.save(model.state_dict(), model_save_path)
            
            if epoch % 100 == 0:
                print(f'Époque: {epoch:03d} | Loss: {loss:.4f} | AUC: {current_auc:.4f} | Best: {best_auc:.4f}')

    return model, best_auc