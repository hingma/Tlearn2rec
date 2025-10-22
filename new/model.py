import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import config

class SimpleGCN(nn.Module):
    """Graph Convolutional Network - Unsupervised version."""

    def __init__(self, in_channels, hidden_channels, embedding_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, embedding_dim)
        self.p = 0.3
        self._printed_shapes = False

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if config.DEBUG_SHAPES and not self._printed_shapes:
            print(f"[SimpleGCN] conv1: in={self.conv1.in_channels}, out={self.conv1.out_channels}")
            print(f"[SimpleGCN] input x: {tuple(x.shape)}, edge_index: {tuple(edge_index.shape)}")
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)

        if config.DEBUG_SHAPES and not self._printed_shapes:
            print(f"[SimpleGCN] conv2: in={self.conv2.in_channels}, out={self.conv2.out_channels}")
            print(f"[SimpleGCN] after conv1: {tuple(x.shape)}")
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)

        if config.DEBUG_SHAPES and not self._printed_shapes:
            print(f"[SimpleGCN] conv3: in={self.conv3.in_channels}, out={self.conv3.out_channels}")
            print(f"[SimpleGCN] after conv2: {tuple(x.shape)}")
        x = self.conv3(x, edge_index)
        embeddings = F.normalize(x, p=2, dim=1)  # L2 normalization for better clustering
        if config.DEBUG_SHAPES and not self._printed_shapes:
            print(f"[SimpleGCN] embeddings: {tuple(embeddings.shape)}")
            self._printed_shapes = True

        return embeddings


class SimpleGAT(nn.Module):
    """Graph Attention Network - Unsupervised version."""

    def __init__(self, in_channels, hidden_channels, embedding_dim=64, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.conv3 = GATConv(hidden_channels * heads, embedding_dim, heads=1, dropout=0.3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        embeddings = F.normalize(x, p=2, dim=1)  # L2 normalization

        return embeddings


class SimpleSAGE(nn.Module):
    """GraphSAGE - Unsupervised version."""

    def __init__(self, in_channels, hidden_channels, embedding_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, embedding_dim)
        self.p = 0.2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.p, training=self.training)

        x = self.conv3(x, edge_index)
        embeddings = F.normalize(x, p=2, dim=1)  # L2 normalization

        return embeddings
