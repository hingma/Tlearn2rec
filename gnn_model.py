# 定义完整的GCN网络，在训练和测试中调用
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
import config


class BipartiteGraphConvolution(MessagePassing):
    def __init__(self, emb_size):
        super().__init__('mean')

        # 处理单个节点的消息传递，将源点，汇点和边的消息拼接后映射为长度emb_size的向量
        self.message_module = nn.Sequential(
            # Input is concatenation of source, edge, and target features
            nn.Linear(3 * emb_size, emb_size),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.post_conv_module = nn.Sequential(
            nn.LayerNorm(emb_size)
        )

        # A module to update the target node's features
        self.output_module = nn.Sequential(
            # Input is concatenation of aggregated messages and old target features
            nn.Linear(2 * emb_size, emb_size),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, source_features, edge_indices, edge_features, target_features):
        # propagate 前后执行消息计算和消息聚合，聚合使用简单的平均方式
        output = self.propagate(edge_indices, size=(source_features.shape[0], target_features.shape[0]),
                                node_features=(source_features, target_features), edge_features=edge_features)
        # 聚合得到的嵌入先进行layernorm归一化，然后和原特征拼接起来
        # 最后经过一个线性层和ReLU，将嵌入再次映射为大小为emb_size的向量
        return self.output_module(torch.cat([self.post_conv_module(output), target_features], dim=-1))

    def message(self, node_features_i, node_features_j, edge_features):
        cat_features = torch.cat([node_features_i, edge_features, node_features_j], dim=-1)
        return self.message_module(cat_features)


class GCNPolicy(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        # Number of features for constraints, edges, and variables.
        cons_nfeats = config.MODEL_PARAMS['cons_nfeats']
        edge_nfeats = config.MODEL_PARAMS['edge_nfeats']
        var_nfeats = config.MODEL_PARAMS['var_nfeats']

        # 整个的GNN网络分为预处理层，GNN层，后处理层
        # ------ Pre-processing layers -------

        # 预处理层负责把约束，节点和边的嵌入映射到相同维度，方便后续输入；
        # 同时多层的MLP可以增强模型的表达能力

        # CONSTRAINT EMBEDDING
        self.cons_embedding = nn.Sequential(
            nn.LayerNorm(cons_nfeats),
            nn.Linear(cons_nfeats, emb_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(emb_size, emb_size),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # EDGE EMBEDDING
        self.edge_embedding = nn.Sequential(
            nn.LayerNorm(edge_nfeats),
            nn.Linear(edge_nfeats, emb_size),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(var_nfeats),
            nn.Linear(var_nfeats, emb_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(emb_size, emb_size),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # ------ GNN layers -------
        # GNN层执行消息传递
        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)

        # ------ Post-processing layers -------
        # 后处理层是为后续对比学习做准备，也可以增强模型表达能力
        self.projection_head = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 128),
        )

    def forward(self, inputs, v_labels=None):
        """
        Accepts a batch of graphs.
        - During training (v_labels is not None), returns projected embeddings and labels for contrastive loss.
        - During inference (v_labels is None), returns final variable embeddings from the GNN.

        Parameters
        ----------
        inputs: list of tensors
            Model input as a bipartite graph. May be batched into a stacked graph.
        v_labels: torch.Tensor, optional
            Variable labels, used to switch between training and inference modes.

        Returns
        -------
        - Training: (projected_embeddings, foreground_labels)
            - projected_embeddings: torch.Tensor, embeddings of foreground variables, passed through the projection head and L2-normalized.
            - foreground_labels: torch.Tensor, labels of the corresponding foreground variables.
        - Inference: variable_features
            - variable_features: torch.Tensor, final embeddings for all variables in the batch.
        """
        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample = inputs
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # 为后续残差连接储存原变量特征
        initial_variable_features = variable_features

        # 进行两轮消息传递，但是参数共享
        n_gnn_layers = 2
        for _ in range(n_gnn_layers):
            constraint_features = self.conv_v_to_c(variable_features, reversed_edge_indices, edge_features, constraint_features)
            variable_features = self.conv_c_to_v(constraint_features, edge_indices, edge_features, variable_features)

        # 添加残差连接，保持训练稳定
        variable_features = variable_features + initial_variable_features

        if v_labels is not None:
            # 训练/验证模式: 返回用于计算对比损失的投影嵌入和标签。
            foreground_mask = v_labels > 0
            
            if not torch.any(foreground_mask):
                return None, None

            fg_embeddings = variable_features[foreground_mask]
            fg_labels = v_labels[foreground_mask]
            
            # 将嵌入通过投影头。这是对比学习的标准做法，
            # 投影头将特征映射到应用对比损失的空间。
            projected_embeddings = self.projection_head(fg_embeddings)
            
            # 对投影后的嵌入进行L2归一化，这是对比损失的标准步骤。
            return F.normalize(projected_embeddings, p=2, dim=1), fg_labels
        else:
            # 推理模式: 直接返回GNN主干网络的最终嵌入。
            # 这是对比学习中的标准实践。研究表明，对于聚类等下游任务，
            # 使用投影头之前的特征（variable_features）通常比使用其之后的特征效果更好，
            # 因为它们包含更通用的信息。
            return variable_features

    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def restore_state(self, path):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
