import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import networkx as nx
#  Visualization Functions

def plot_embeddings_and_clusters(embeddings, clustering_results, true_labels, data):
    """Visualize embeddings and clustering results."""
    # Convert embeddings to numpy
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Reduce dimensionality for visualization
    if embeddings_np.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_np)
    else:
        embeddings_2d = embeddings_np
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Clustering Results Comparison', fontsize=16)
    
    # Ground truth visualization
    axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      c=true_labels, cmap='Set1', alpha=0.7)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].set_xlabel('PC1' if embeddings_np.shape[1] > 2 else 'Dim 1')
    axes[0, 0].set_ylabel('PC2' if embeddings_np.shape[1] > 2 else 'Dim 2')
    
    # K-means results
    axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      c=clustering_results['kmeans'], cmap='Set1', alpha=0.7)
    axes[0, 1].set_title('K-Means Clustering')
    axes[0, 1].set_xlabel('PC1' if embeddings_np.shape[1] > 2 else 'Dim 1')
    axes[0, 1].set_ylabel('PC2' if embeddings_np.shape[1] > 2 else 'Dim 2')
    
    # Spectral clustering results
    axes[1, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      c=clustering_results['spectral'], cmap='Set1', alpha=0.7)
    axes[1, 0].set_title('Spectral Clustering')
    axes[1, 0].set_xlabel('PC1' if embeddings_np.shape[1] > 2 else 'Dim 1')
    axes[1, 0].set_ylabel('PC2' if embeddings_np.shape[1] > 2 else 'Dim 2')
    
    # Hierarchical clustering results
    axes[1, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      c=clustering_results['hierarchical'], cmap='Set1', alpha=0.7)
    axes[1, 1].set_title('Hierarchical Clustering')
    axes[1, 1].set_xlabel('PC1' if embeddings_np.shape[1] > 2 else 'Dim 1')
    axes[1, 1].set_ylabel('PC2' if embeddings_np.shape[1] > 2 else 'Dim 2')
    
    plt.tight_layout()
    plt.show()

def plot_network_clusters(data, clustering_results, true_labels):
    """Plot the network with different clustering results."""
    # Convert to NetworkX graph
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    G.add_edges_from(edge_index.T)
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Network Clustering Results', fontsize=16)
    
    # Common layout for all plots
    pos = nx.spring_layout(G, seed=42)
    
    # Ground truth
    nx.draw(G, pos, ax=axes[0, 0], node_color=true_labels, 
           cmap=plt.cm.Set1, with_labels=True, node_size=200, 
           font_size=6, font_color='white')
    axes[0, 0].set_title('Ground Truth Communities')
    
    # K-means clustering
    nx.draw(G, pos, ax=axes[0, 1], node_color=clustering_results['kmeans'], 
           cmap=plt.cm.Set1, with_labels=True, node_size=200, 
           font_size=6, font_color='white')
    axes[0, 1].set_title('K-Means Clustering')
    
    # Spectral clustering
    nx.draw(G, pos, ax=axes[1, 0], node_color=clustering_results['spectral'], 
           cmap=plt.cm.Set1, with_labels=True, node_size=200, 
           font_size=6, font_color='white')
    axes[1, 0].set_title('Spectral Clustering')
    
    # Hierarchical clustering
    nx.draw(G, pos, ax=axes[1, 1], node_color=clustering_results['hierarchical'], 
           cmap=plt.cm.Set1, with_labels=True, node_size=200, 
           font_size=6, font_color='white')
    axes[1, 1].set_title('Hierarchical Clustering')
    
    plt.tight_layout()
    plt.show()

def plot_training_loss(losses, title="Training Loss"):
    """Plot training loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
