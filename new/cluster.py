import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Clustering Algorithms

class GraphClustering:
    """Collection of clustering algorithms for graph embeddings."""
    
    def __init__(self, n_clusters, use_gpu=False):
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu
        
    def kmeans_clustering(self, embeddings):
        """K-means clustering on embeddings."""
        if self.use_gpu and torch.is_tensor(embeddings) and embeddings.is_cuda:
            return self.gpu_kmeans_clustering(embeddings)
        else:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            # Ensure embeddings are on CPU and converted to numpy
            if torch.is_tensor(embeddings):
                embeddings_np = embeddings.detach().cpu().numpy()
            else:
                embeddings_np = embeddings
            cluster_labels = kmeans.fit_predict(embeddings_np)
            return cluster_labels, kmeans
    
    def spectral_clustering(self, embeddings):
        """Spectral clustering on embeddings."""
        if self.use_gpu and torch.is_tensor(embeddings) and embeddings.is_cuda:
            return self.gpu_spectral_clustering(embeddings)
        else:
            spectral = SpectralClustering(
                n_clusters=self.n_clusters, 
                random_state=42,
                affinity='nearest_neighbors'
            )
            # Ensure embeddings are on CPU and converted to numpy
            if torch.is_tensor(embeddings):
                embeddings_np = embeddings.detach().cpu().numpy()
            else:
                embeddings_np = embeddings
            cluster_labels = spectral.fit_predict(embeddings_np)
            return cluster_labels, spectral
    
    def hierarchical_clustering(self, embeddings):
        """Agglomerative hierarchical clustering."""
        if self.use_gpu and torch.is_tensor(embeddings) and embeddings.is_cuda:
            return self.gpu_hierarchical_clustering(embeddings)
        else:
            hierarchical = AgglomerativeClustering(n_clusters=self.n_clusters)
            # Ensure embeddings are on CPU and converted to numpy
            if torch.is_tensor(embeddings):
                embeddings_np = embeddings.detach().cpu().numpy()
            else:
                embeddings_np = embeddings
            cluster_labels = hierarchical.fit_predict(embeddings_np)
            return cluster_labels, hierarchical

    # GPU-native PyTorch implementations
    def gpu_kmeans_clustering(self, embeddings, max_iters=100, tol=1e-4):
        """GPU-native K-means clustering using PyTorch."""
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings, device='cuda')
        
        n, d = embeddings.shape
        device = embeddings.device
        
        # Initialize centroids using k-means++
        centroids = self._kmeans_plus_plus_init(embeddings, self.n_clusters)
        
        prev_loss = float('inf')
        for iteration in range(max_iters):
            # Compute distances to centroids
            distances = torch.cdist(embeddings, centroids)  # [n, k]
            labels = torch.argmin(distances, dim=1)  # [n]
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = embeddings[mask].mean(0)
                else:
                    # Handle empty clusters by reinitializing
                    new_centroids[k] = embeddings[torch.randint(0, n, (1,))]
            
            # Check convergence
            loss = torch.sum((embeddings - centroids[labels]) ** 2)
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
            centroids = new_centroids
        
        return labels.cpu().numpy(), centroids

    def gpu_spectral_clustering(self, embeddings, n_neighbors=10):
        """GPU-native spectral clustering using PyTorch."""
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings, device='cuda')
        
        n = embeddings.shape[0]
        device = embeddings.device
        
        # Build k-NN graph
        distances = torch.cdist(embeddings, embeddings)
        
        # Get k nearest neighbors for each point
        _, knn_indices = torch.topk(distances, k=min(n_neighbors + 1, n), dim=1, largest=False)
        
        # Build adjacency matrix
        adjacency = torch.zeros(n, n, device=device)
        for i in range(n):
            for j in knn_indices[i, 1:]:  # Skip self (index 0)
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0  # Make symmetric
        
        # Compute normalized Laplacian
        degree = adjacency.sum(dim=1)
        degree_sqrt_inv = torch.where(degree > 0, 1.0 / torch.sqrt(degree), torch.zeros_like(degree))
        laplacian = torch.eye(n, device=device) - torch.outer(degree_sqrt_inv, degree_sqrt_inv) * adjacency
        
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        
        # Use smallest k eigenvectors
        embedding_spectral = eigenvectors[:, :self.n_clusters]
        
        # Normalize rows
        embedding_spectral = F.normalize(embedding_spectral, p=2, dim=1)
        
        # Apply k-means to spectral embedding
        labels, _ = self.gpu_kmeans_clustering(embedding_spectral)
        
        return labels, embedding_spectral

    def gpu_hierarchical_clustering(self, embeddings):
        """GPU-native agglomerative clustering using PyTorch."""
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings, device='cuda')
        
        n = embeddings.shape[0]
        device = embeddings.device
        
        # Initialize each point as its own cluster
        labels = torch.arange(n, device=device)
        active_clusters = torch.ones(n, dtype=torch.bool, device=device)
        
        # Compute initial distance matrix
        distances = torch.cdist(embeddings, embeddings)
        distances.fill_diagonal_(float('inf'))  # Ignore self-distances
        
        n_clusters_current = n
        
        while n_clusters_current > self.n_clusters:
            # Find closest pair of clusters
            active_mask = active_clusters.unsqueeze(0) & active_clusters.unsqueeze(1)
            masked_distances = torch.where(active_mask, distances, torch.tensor(float('inf'), device=device))
            
            min_idx = torch.argmin(masked_distances)
            i, j = min_idx // n, min_idx % n
            
            # Merge clusters i and j (merge j into i)
            labels[labels == j] = i
            active_clusters[j] = False
            
            # Update distances (single linkage)
            for k in range(n):
                if active_clusters[k] and k != i:
                    distances[i, k] = distances[k, i] = torch.min(distances[i, k], distances[j, k])
            
            n_clusters_current -= 1
        
        # Relabel to consecutive integers
        unique_labels = torch.unique(labels)
        label_map = {old.item(): new for new, old in enumerate(unique_labels)}
        final_labels = torch.tensor([label_map[l.item()] for l in labels], device=device)
        
        return final_labels.cpu().numpy(), None

    def _kmeans_plus_plus_init(self, embeddings, k):
        """K-means++ initialization for better convergence."""
        n, d = embeddings.shape
        device = embeddings.device
        
        centroids = torch.zeros(k, d, device=device)
        
        # Choose first centroid randomly
        centroids[0] = embeddings[torch.randint(0, n, (1,))]
        
        for i in range(1, k):
            # Compute distances to nearest centroid
            distances = torch.cdist(embeddings, centroids[:i])
            min_distances = torch.min(distances, dim=1)[0]
            
            # Choose next centroid with probability proportional to squared distance
            probs = min_distances ** 2
            probs = probs / probs.sum()
            
            # Sample according to probabilities
            cumprobs = torch.cumsum(probs, dim=0)
            r = torch.rand(1, device=device)
            idx = torch.searchsorted(cumprobs, r)
            centroids[i] = embeddings[idx]
        
        return centroids

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
