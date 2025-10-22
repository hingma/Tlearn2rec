import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from torch_geometric.utils import to_undirected

import config
from datasets import build_loaders, load_karate
from model import SimpleGCN, SimpleGAT, SimpleSAGE
from visualize import plot_embeddings_and_clusters, plot_network_clusters, plot_training_loss, plot_validation_loss, plot_karate_score  


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        pos = F.cosine_similarity(embeddings[row], embeddings[col], dim=1)
        pos = torch.exp(pos / self.temperature)

        num_nodes = embeddings.size(0)
        neg_idx = torch.randint(0, num_nodes, (row.numel(),), device=embeddings.device)
        neg = F.cosine_similarity(embeddings[col], embeddings[neg_idx], dim=1)
        neg = torch.exp(neg / self.temperature)

        loss = -torch.log(pos / (pos + neg + 1e-12)).mean()
        return loss


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Adam, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_graphs = 0

    for data in loader:
        data = data.to(device)
        # Debug shapes
        if getattr(config, 'DEBUG_SHAPES', False) and not getattr(model, '_printed_train_batch', False):
            print(f"[train] batch x: {tuple(data.x.shape)} | edge_index: {tuple(data.edge_index.shape)}")
            if hasattr(data, 'batch') and data.batch is not None:
                print(f"[train] batch vector: {tuple(data.batch.shape)}")
            model._printed_train_batch = True
        #===============================================
        optimizer.zero_grad()
        embeddings = model(data)
        edge_index = to_undirected(data.edge_index)
        loss = criterion(embeddings, edge_index)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_graphs += 1

    return total_loss / max(1, total_graphs)


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    for data in loader:
        data = data.to(device)
        if getattr(config, 'DEBUG_SHAPES', False) and not getattr(model, '_printed_valid_batch', False):
            print(f"[valid] batch x: {tuple(data.x.shape)} | edge_index: {tuple(data.edge_index.shape)}")
            model._printed_valid_batch = True
        embeddings = model(data)
        loss = criterion(embeddings, to_undirected(data.edge_index))
        total_loss += loss.item()
        total_graphs += 1
    return total_loss / max(1, total_graphs)


@torch.no_grad()
def eval_on_karate(model: nn.Module, device: torch.device) -> float:
    data = load_karate()
    data = data.to(device)
    # Ensure x has the expected feature dimension for the current model
    in_channels_expected = getattr(model, 'conv1').in_channels
    x = data.x
    if x is None:
        x = torch.eye(data.num_nodes, device=device)
    if x.size(-1) < in_channels_expected:
        # pad with zeros
        pad = in_channels_expected - x.size(-1)
        x = F.pad(x, (0, pad))
    elif x.size(-1) > in_channels_expected:
        # project with a fixed random matrix (deterministic by seed)
        torch.manual_seed(config.SEED)
        proj = torch.randn(x.size(-1), in_channels_expected, device=device)
        x = x @ proj
    data.x = x
    embeddings = model(data)
    # simple proxy: intra-edge cosine similarity mean (higher is better)
    row, col = data.edge_index
    sim = F.cosine_similarity(embeddings[row], embeddings[col], dim=1)
    return sim.mean().item()


def main():
    torch.manual_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader = build_loaders()

    # Infer in_channels from first training graph
    sample = next(iter(train_loader))
    in_channels = sample.x.size(-1)

    if getattr(config, 'DEBUG_SHAPES', False):
        print(f"[setup] inferred in_channels from dataset: {in_channels}")
        print(f"[setup] sample x: {tuple(sample.x.shape)}, edge_index: {tuple(sample.edge_index.shape)}")

    for model in [SimpleGAT, SimpleSAGE, SimpleGCN]:
        model = model(in_channels=in_channels, hidden_channels=config.HIDDEN_DIM, embedding_dim=config.EMBED_DIM).to(device)
        optimizer = Adam(model.parameters(), lr=config.LR)
        criterion = ContrastiveLoss(temperature=config.TEMPERATURE)

        best_val = float('inf')
        epochs_no_improve = 0
        best_path = config.EXPERIMENT_DIR / f'best_{model.__class__.__name__}.pt'

        for epoch in range(1, config.MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = eval_loss(model, valid_loader, criterion, device)
            karate_score = eval_on_karate(model, device)

            print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | karate {karate_score:.4f}")

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save({'state_dict': model.state_dict(), 'in_channels': in_channels}, best_path)

        print(f"Best {model.__class__.__name__} model saved to: {best_path}")

        plot_training_loss(train_loss)
        plot_validation_loss(val_loss)
        plot_karate_score(karate_score)

if __name__ == '__main__':
    main()


