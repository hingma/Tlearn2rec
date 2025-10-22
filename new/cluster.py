import torch
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Clustering Algorithms

class GraphClustering:
    """Collection of clustering algorithms for graph embeddings."""
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        
    def kmeans_clustering(self, embeddings):
        """K-means clustering on embeddings."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings.detach().cpu().numpy())
        return cluster_labels, kmeans
    
    def spectral_clustering(self, embeddings):
        """Spectral clustering on embeddings."""
        spectral = SpectralClustering(
            n_clusters=self.n_clusters, 
            random_state=42,
            affinity='nearest_neighbors'
        )
        cluster_labels = spectral.fit_predict(embeddings.detach().cpu().numpy())
        return cluster_labels, spectral
    
    def hierarchical_clustering(self, embeddings):
        """Agglomerative hierarchical clustering."""
        hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
        cluster_labels = hierarchical.fit_predict(embeddings.detach().cpu().numpy())
        return cluster_labels, hierarchical

class ClusteringEvaluator:
    """Evaluation metrics for clustering performance."""
    
    @staticmethod
    def evaluate_clustering(true_labels, pred_labels, embeddings):
        """Comprehensive clustering evaluation."""
        results = {}
        
        # External metrics (need ground truth)
        if true_labels is not None:
            results['ARI'] = adjusted_rand_score(true_labels, pred_labels)
            results['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
        
        # Internal metrics (unsupervised)
        embeddings_np = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else embeddings
        results['Silhouette'] = silhouette_score(embeddings_np, pred_labels)
        results['Calinski_Harabasz'] = calinski_harabasz_score(embeddings_np, pred_labels)
        results['Davies_Bouldin'] = davies_bouldin_score(embeddings_np, pred_labels)
        
        return results
    
    @staticmethod
    def print_results(results, method_name):
        """Pretty print clustering results."""
        print(f"\n=== {method_name} Results ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
