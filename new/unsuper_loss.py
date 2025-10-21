import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAutoEncoder(nn.Module):
    """Graph Autoencoder for unsupervised graph embedding learning."""
    
    def __init__(self, encoder, decoder_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(encoder.conv3.out_channels, decoder_dim)
        
    def forward(self, data):
        # Encode
        embeddings = self.encoder(data)
        # Decode (reconstruct adjacency or features)
        reconstructed = torch.sigmoid(self.decoder(embeddings))
        return embeddings, reconstructed

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
    def reconstruction_loss(original_features, reconstructed_features):
        """MSE reconstruction loss."""
        return F.mse_loss(reconstructed_features, original_features)
    
    @staticmethod
    def proximity_loss(embeddings, edge_index):
        """Encourage connected nodes to have similar embeddings."""
        row, col = edge_index
        connected_embeddings = embeddings[row]
        target_embeddings = embeddings[col]
        
        # L2 distance between connected nodes
        proximity = F.mse_loss(connected_embeddings, target_embeddings)
        return proximity

print("✓ Unsupervised loss functions defined")
