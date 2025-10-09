# 读取data/processed/[problem]/train文件，训练GNN模型
# 将最好的模型保存到model/[problem]/GCNPolicy/best_params.pkl
import os
import argparse
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from gnn_model import GCNPolicy
from utilities import log
import config

class MILPDataset(Dataset):
    """
    A PyTorch dataset for loading MILP samples from .pt files.
    """
    def __init__(self, sample_files):
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        data = torch.load(self.sample_files[idx], weights_only=False)
        return data


def collate_fn(samples):
    """
    A custom collate function to batch MILP graph samples (HeteroData) for metric learning.
    This version uses PyTorch operations to avoid NumPy overhead and offsets labels
    to ensure their uniqueness across different samples in a batch.
    """
    # Get num_classes from config
    num_classes = config.TRAIN_PARAMS['num_classes']
    c_features_list, e_indices_list, e_features_list, v_features_list, v_labels_list = [], [], [], [], []
    n_cs_list, n_vs_list = [], []

    c_offset, v_offset, label_offset = 0, 0, 0
    num_classes_per_sample = num_classes

    for sample in samples:
        # sample is a HeteroData object, extract tensors directly
        c_features = sample['constraint'].x
        v_features = sample['variable'].x
        # The model expects a single edge type, from constraint to variable
        e_indices = sample['constraint', 'includes', 'variable'].edge_index
        e_features = sample['constraint', 'includes', 'variable'].edge_attr
        # Clone to avoid modifying the original data in the dataset cache
        v_labels = sample['variable'].y.clone()

        num_c, num_v = c_features.shape[0], v_features.shape[0]

        # --- Label Offsetting ---
        # Get a mask for foreground variables (labels > 0)
        foreground_mask = v_labels > 0
        # Add the offset to make labels unique across the batch.
        # This prevents the contrastive loss from incorrectly pairing variables from different graphs.
        if torch.any(foreground_mask):
            v_labels[foreground_mask] += label_offset
        # --- End Label Offsetting ---

        c_features_list.append(c_features)
        v_features_list.append(v_features)
        e_features_list.append(e_features)
        v_labels_list.append(v_labels)

        # The edge indices need to be offset for batching
        e_indices_list.append(e_indices + torch.tensor([[c_offset], [v_offset]], dtype=torch.long))

        n_cs_list.append(num_c)
        n_vs_list.append(num_v)

        c_offset += num_c
        v_offset += num_v
        # Increment the offset for the next sample in the batch
        label_offset += num_classes_per_sample

    # Concatenate lists of tensors
    c_features = torch.cat(c_features_list, dim=0).float()
    e_indices = torch.cat(e_indices_list, dim=1)
    e_features = torch.cat(e_features_list, dim=0).float()
    v_features = torch.cat(v_features_list, dim=0).float()
    v_labels = torch.cat(v_labels_list, dim=0)

    n_cs = torch.tensor(n_cs_list, dtype=torch.int32)
    n_vs = torch.tensor(n_vs_list, dtype=torch.int32)

    return c_features, e_indices, e_features, v_features, v_labels, n_cs, n_vs


class SupervisedContrastiveLoss(nn.Module):
    """
    Implementation of the Supervised Contrastive Learning loss function from https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # embeddings: (N, D), L2-normalized
        # labels: (N,)
        
        # Create a mask for positive pairs (samples with the same label)
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        # Mask out self-comparisons
        labels_matrix.fill_diagonal_(False)
        
        # If there are no positive pairs in the batch, loss is 0.
        if not labels_matrix.any():
            return torch.tensor(0.0, device=embeddings.device)

        # Compute cosine similarity between all pairs of embeddings.
        sim_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Mask to exclude self-similarity from the denominator's log-sum-exp.
        logits_mask = torch.ones_like(sim_matrix).fill_diagonal_(0)
        
        # For each anchor, the loss is computed over its positive pairs.
        # log_prob = sim_ij/T - log(sum_{k!=i} exp(sim_ik/T))
        exp_sim = torch.exp(sim_matrix / self.temperature)
        log_prob = (sim_matrix / self.temperature) - torch.log((exp_sim * logits_mask).sum(1, keepdim=True))
        
        # Compute the mean of log-likelihood over all positive pairs for each anchor.
        mean_log_prob_pos = (labels_matrix * log_prob).sum(1) / labels_matrix.sum(1).clamp(min=1)
        
        # The final loss is the negative of this mean log-likelihood, averaged over all anchors
        # that have at least one positive pair.
        loss = -mean_log_prob_pos
        
        has_positives = labels_matrix.sum(1) > 0
        loss = loss[has_positives].mean()
        
        return loss


def process(model, dataloader, criterion, optimizer=None, scaler=None, device='cpu', epoch=None, phase='train'):
    mean_loss = 0
    n_samples_processed = 0
    is_train = optimizer is not None

    if is_train: # 训练模式
        model.train()
    else:
        model.eval()

    for batch in dataloader:
        c, ei, ev, v, v_labels, n_cs, n_vs = [
            t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
        ]
        batch_size = len(n_cs)

        model_input = (c, ei, ev, v, n_cs, n_vs)

        if is_train:
            optimizer.zero_grad()

        # 使用 autocast 上下文管理器进行混合精度前向传播
        with autocast(device_type="cuda", enabled=(scaler is not None)):
            # Model now returns projected embeddings and their labels for foreground variables
            proj_embeddings, fg_labels = model(model_input, v_labels)

            if proj_embeddings is not None and fg_labels is not None:
                loss = criterion(proj_embeddings, fg_labels)
            else:
                # 批次中没有前景变量或正样本对
                loss = torch.tensor(0.0, device=device)
        
        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # 如果不使用混合精度（例如在CPU上）
                loss.backward()
                optimizer.step()

        mean_loss += loss.item() * batch_size
        n_samples_processed += batch_size

    if n_samples_processed > 0:
        mean_loss /= n_samples_processed

    return mean_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['facilities', 'osif'],
    )
 
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    # 从配置文件加载训练参数
    train_params = config.TRAIN_PARAMS
    model_params = config.MODEL_PARAMS
    seed = train_params['seed']
    max_epochs = train_params['max_epochs']
    batch_size = train_params['batch_size']
    valid_batch_size = train_params['valid_batch_size']
    lr = train_params['lr']
    patience = train_params['patience']
    early_stopping = train_params['early_stopping']
    emb_size = model_params['emb_size']
    num_classes = train_params['num_classes']

    running_dir = config.MODELS_DIR / args.problem / "GCNPolicy"

    os.makedirs(running_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 初始化 GradScaler 用于混合精度训练
    scaler = None
    if device.type == 'cuda':
        scaler = GradScaler(device='cuda')
        print("Using mixed precision training with GradScaler.")


    ### LOG ###
    logfile = running_dir / 'log.txt'
    log(f"Log file at: {logfile}", logfile)
    log(f"Problem: {args.problem}", logfile)
    log(f"Device: {device}", logfile)
    log(f"Training parameters: {train_params}", logfile)
    log(f"Model parameters: {model_params}", logfile)

    rng = np.random.RandomState(seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))

    ### SET-UP DATASET ###
    train_files = list((config.PROCESSED_DATA_DIR / args.problem / 'train').glob('*.pt'))
    valid_files = list((config.PROCESSED_DATA_DIR / args.problem / 'valid').glob('*.pt'))

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    train_dataset = MILPDataset(train_files)
    valid_dataset = MILPDataset(valid_files)

    # Create DataLoaders once, outside the loop, for efficiency.
    # Create DataLoader with a fixed num_classes for collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            collate_fn=collate_fn, num_workers=train_params['num_workers'], 
                            pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=train_params['num_workers'], 
                            pin_memory=True)

    ### MODEL LOADING ###
    model = GCNPolicy(emb_size=emb_size)
    model.to(device)

    ### TRAINING LOOP ###
    criterion = SupervisedContrastiveLoss(temperature=train_params['temperature']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = np.inf
    plateau_count = 0

    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        # TRAIN
        train_loss = process(model, train_loader, criterion, optimizer, scaler, device, epoch, 'train')
        log(f"TRAIN LOSS: {train_loss:0.3f}", logfile)

        # VALIDATE
        valid_loss = process(model, valid_loader, criterion, None, scaler, device, epoch, 'valid')
        log(f"VALID LOSS: {valid_loss:0.3f}", logfile)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(running_dir / 'best_params.pkl')
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count >= early_stopping:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            # Reduce learning rate on plateau
            if plateau_count % patience == 0:
                lr *= 0.2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr:.1e}", logfile)

    model.restore_state(running_dir / 'best_params.pkl')
    valid_loss = process(model, valid_loader, criterion, None, scaler, device)
    log(f"BEST VALID LOSS: {valid_loss:0.3f}", logfile)
