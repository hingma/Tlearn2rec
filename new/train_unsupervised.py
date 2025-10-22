import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from torch_geometric.utils import to_undirected

import config
from datasets import build_loaders, load_karate
from model import SimpleGCN, SimpleGAT, SimpleSAGE
from visualize import plot_embeddings_and_clusters, plot_network_clusters, plot_training_loss, plot_validation_loss, plot_karate_score  
from cluster import GraphClustering, ClusteringEvaluator



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


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Optimizer, device: torch.device) -> float:
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


@torch.no_grad()
def run_clustering_pipeline(embedding: torch.Tensor,
                            n_clusters: int | None = None,
                            true_labels: torch.Tensor | None = None,
                            use_gpu: bool = False):
    """Mirror the notebook pipeline: compute embeddings, run 3 clustering methods, evaluate.

    Returns dict with embeddings, predicted labels per method, and evaluation metrics per method.
    """
    if n_clusters is None and true_labels is not None:
        unique = torch.unique(true_labels)
        n_clusters = int(unique.numel())
    if n_clusters is None:
        n_clusters = 2

    clusterer = GraphClustering(n_clusters=n_clusters, use_gpu=use_gpu)
    evaluator = ClusteringEvaluator()

    results_summary = {}
    clustering_results = {}

    # K-Means
    try:
        kmeans_labels, _ = clusterer.kmeans_clustering(embedding)
        kmeans_results = evaluator.evaluate_clustering(true_labels, kmeans_labels, embedding)
        # evaluator.print_results(kmeans_results, "K-Means")
        results_summary['K-Means'] = kmeans_results
        clustering_results['kmeans'] = kmeans_labels
    except Exception as e:
        print(f"[clustering] K-Means failed: {e}")

    # Spectral
    try:
        spectral_labels, _ = clusterer.spectral_clustering(embedding)
        spectral_results = evaluator.evaluate_clustering(true_labels, spectral_labels, embedding)
        # evaluator.print_results(spectral_results, "Spectral")
        results_summary['Spectral'] = spectral_results
        clustering_results['spectral'] = spectral_labels
    except Exception as e:
        print(f"[clustering] Spectral failed: {e}")

    # Hierarchical
    try:
        hierarchical_labels, _ = clusterer.hierarchical_clustering(embedding)
        hierarchical_results = evaluator.evaluate_clustering(true_labels, hierarchical_labels, embedding)
        # evaluator.print_results(hierarchical_results, "Hierarchical")
        results_summary['Hierarchical'] = hierarchical_results
        clustering_results['hierarchical'] = hierarchical_labels
    except Exception as e:
        print(f"[clustering] Hierarchical failed: {e}")

    return {
        'embeddings': embedding,
        'clustering_results': clustering_results,
        'evaluation_results': results_summary,
    }


@torch.no_grad()
def evaluate_clustering_on_loader(model: nn.Module,
                                  loader: DataLoader,
                                  device: torch.device) -> None:
    """Evaluate clustering quality per-graph in a (possibly batched) loader and print averages.

    - Uses true labels `y` when available to compute ARI/NMI; always computes internal metrics.
    - Number of clusters is inferred from true labels when available; otherwise defaults to 2.
    """
    model.eval()

    # Aggregators per method -> metric -> list of values
    agg: dict[str, dict[str, list[float]]] = {}

    for batch in loader:
        batch = batch.to(device)
        embeddings = model(batch)

        # Determine graph ids in this batch (PyG provides `batch` vector when multiple graphs)
        if hasattr(batch, 'batch') and batch.batch is not None:
            graph_ids = torch.unique(batch.batch).tolist()
        else:
            graph_ids = [None]

        for gid in graph_ids:
            if gid is None:
                node_mask = torch.ones(embeddings.size(0), dtype=torch.bool, device=device)
            else:
                node_mask = (batch.batch == gid)

            emb_g = embeddings[node_mask]
            y_g = batch.y[node_mask] if (hasattr(batch, 'y') and batch.y is not None) else None

            if emb_g.size(0) < 2:
                continue

            n_clusters = int(torch.unique(y_g).numel()) if y_g is not None else 2

            out = run_clustering_pipeline(embedding=emb_g,  # bypass model; we already have emb_g
                                          n_clusters=n_clusters,
                                          true_labels=y_g,
                                          use_gpu=device.type == 'cuda')

            # Collect metrics
            for method_name, metrics in out['evaluation_results'].items():
                if method_name not in agg:
                    agg[method_name] = {k: [] for k in metrics.keys()}
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        agg[method_name][metric].append(value)

    # Print averages
    if not agg:
        print("[clustering] No metrics collected.")
        return

    print("\n=== Clustering evaluation (averaged over validation graphs) ===")
    for method_name, metrics in agg.items():
        print(f"\n-- {method_name} --")
        for metric, values in metrics.items():
            if len(values) == 0:
                continue
            mean_val = sum(values) / len(values)
            print(f"{metric}: {mean_val:.4f}")

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
        train_losses = []
        val_losses = []
        karate_scores = []
        for epoch in range(1, config.MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = eval_loss(model, valid_loader, criterion, device)
            karate_score = eval_on_karate(model, device)
            #
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            karate_scores.append(karate_score)
            #
            print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | karate {karate_score:.4f}")
            #
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save({'state_dict': model.state_dict(), 'in_channels': in_channels}, best_path)

        print(f"Best {model.__class__.__name__} model saved to: {best_path}")

        plot_training_loss(train_losses)
        plot_validation_loss(val_losses)
        plot_karate_score(karate_scores)

        # Clustering-based evaluation on validation set using ground-truth labels when available
        evaluate_clustering_on_loader(model, valid_loader, device)

if __name__ == '__main__':
    main()


