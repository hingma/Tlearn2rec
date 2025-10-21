#!/usr/bin/env python3
"""
Debug file for isolating and debugging the optimizer.zero_grad() issue
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import your model
import sys
sys.path.append('/Users/mox/Research/MILP/Tlearn2rec/new')
from model import SimpleGAT, SimpleGCN, SimpleSAGE

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class UnsupervisedLoss:
    """Collection of unsupervised loss functions for graph learning."""
    
    @staticmethod
    def contrastive_loss(embeddings, edge_index, temperature=0.1):
        """InfoNCE-style contrastive loss for connected nodes."""
        # Create positive pairs from edges
        row, col = edge_index
        pos_embeddings = embeddings[row]
        anchor_embeddings = embeddings[col]
        
        # Compute similarity scores
        pos_sim = F.cosine_similarity(pos_embeddings, anchor_embeddings, dim=1)
        pos_sim = torch.exp(pos_sim / temperature)
        
        # Negative sampling - random node pairs
        num_nodes = embeddings.size(0)
        neg_indices = torch.randint(0, num_nodes, (len(row),), device=embeddings.device)
        neg_embeddings = embeddings[neg_indices]
        neg_sim = F.cosine_similarity(anchor_embeddings, neg_embeddings, dim=1)
        neg_sim = torch.exp(neg_sim / temperature)
        
        # InfoNCE loss
        loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        return loss
    
    @staticmethod
    def proximity_loss(embeddings, edge_index):
        """Encourage connected nodes to have similar embeddings."""
        row, col = edge_index
        connected_embeddings = embeddings[row]
        target_embeddings = embeddings[col]
        
        # L2 distance between connected nodes
        proximity = F.mse_loss(connected_embeddings, target_embeddings)
        return proximity

# Complete Example: Unsupervised Graph Clustering

def run_complete_clustering_example():
    """Run the complete unsupervised clustering example."""
    print("🚀 Starting Unsupervised Graph Clustering Experiment")
    print("="*70)
    
    # Load data
    dataset = KarateClub()
    data = dataset[0]
    true_labels = data.y.numpy()
    n_clusters = len(np.unique(true_labels))
    
    print(f"Dataset: Karate Club")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Features: {data.num_node_features}")
    print(f"True clusters: {n_clusters}")
    
    # Model configurations
    models_config = {
        'GCN': SimpleGCN(data.num_node_features, 32, 16),
        'GAT': SimpleGAT(data.num_node_features, 16, 16, heads=2),
        'SAGE': SimpleSAGE(data.num_node_features, 32, 16)
    }
    
    all_results = {}
    
    # Test different approaches
    for model_name, model in models_config.items():
        print(f"\n{'='*70}")
        print(f"🔬 Testing {model_name} with different training approaches")
        print('='*70)
        
        # 1. Contrastive Learning Approach
        print(f"\n📊 {model_name} - Contrastive Learning")
        print("-" * 50)
        
        model_copy = type(model)(data.num_node_features, 
                               model.conv1.out_channels if hasattr(model.conv1, 'out_channels') else 32, 
                               16)
        trained_model, losses = train_unsupervised_gnn(
            model_copy, data, epochs=100, lr=0.01, loss_type='contrastive'
        )
        
        # Plot training loss
        plot_training_loss(losses, f"{model_name} - Contrastive Learning Loss")
        
        # Run clustering
        results = run_clustering_pipeline(trained_model, data, n_clusters, true_labels)
        
        # Visualizations
        plot_embeddings_and_clusters(
            results['embeddings'], 
            results['clustering_results'], 
            true_labels, data
        )
        plot_network_clusters(data, results['clustering_results'], true_labels)
        
        all_results[f'{model_name}_contrastive'] = results
        
        # 2. Proximity Learning Approach
        print(f"\n📊 {model_name} - Proximity Learning")
        print("-" * 50)
        
        model_copy2 = type(model)(data.num_node_features, 
                                model.conv1.out_channels if hasattr(model.conv1, 'out_channels') else 32, 
                                16)
        trained_model2, losses2 = train_unsupervised_gnn(
            model_copy2, data, epochs=100, lr=0.01, loss_type='proximity'
        )
        
        # Plot training loss
        plot_training_loss(losses2, f"{model_name} - Proximity Learning Loss")
        
        # Run clustering
        results2 = run_clustering_pipeline(trained_model2, data, n_clusters, true_labels)
        
        # Visualizations
        plot_embeddings_and_clusters(
            results2['embeddings'], 
            results2['clustering_results'], 
            true_labels, data
        )
        
        all_results[f'{model_name}_proximity'] = results2
    
    # Summary comparison
    print("\n" + "="*80)
    print("📋 FINAL RESULTS SUMMARY")
    print("="*80)
    
    summary_metrics = ['ARI', 'NMI', 'Silhouette']
    
    for metric in summary_metrics:
        print(f"\n🎯 {metric} Scores:")
        print("-" * 50)
        for exp_name, results in all_results.items():
            best_clustering = max(results['evaluation_results'].items(), 
                                key=lambda x: x[1].get(metric, -1))
            best_method, best_score = best_clustering
            print(f"{exp_name:25} | {best_method:12} | {best_score[metric]:.4f}")
    
    return all_results

# Run the complete example
print("✓ Complete clustering example ready to run!")

if __name__ == "__main__":
    run_complete_clustering_example()
