# 测试给定文件夹中的算例，生成测试报告，并将每个算例的聚类结果保存为JSON格式，
# 储存在result/[problem]/GCNPolicy/cluster_outputs
import collections
import json
import os
import argparse
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from tqdm import tqdm
from gnn_model import GCNPolicy
from utilities import log, hierarchical_dbscan
import config

class MILPDataset(Dataset):
    """A simple dataset class for loading .pt files."""
    def __init__(self, sample_files):
        self.sample_files = sample_files

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        # 同时返回数据和文件路径，以便在评估循环中使用
        return torch.load(self.sample_files[idx], weights_only=False), self.sample_files[idx]


def simple_collate(batch):
    """
    A simple collate function for use when batch_size is 1.
    The DataLoader wraps each sample in a list, so this function simply extracts
    the single sample from the list. This is necessary because the default collate
    function cannot handle HeteroData objects.
    """
    assert len(batch) == 1, "This collate function is designed for batch_size=1 only."
    # batch[0] 是一个 (HeteroData, filepath) 元组
    return batch[0]


def evaluate(model, dataloader, device, true_n_clusters, logfile):
    """
    Evaluates the GNN model by extracting embeddings and calculating clustering metrics.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    log(f"Saving cluster outputs to: {output_dir}", logfile)

    all_embeddings_list = []
    all_labels_list = []

    # Lists to store per-instance metrics
    per_instance_ari = []
    per_instance_nmi = []
    per_instance_v_measure = []

    per_instance_dbscan_ari = []
    per_instance_dbscan_nmi = []
    per_instance_dbscan_v_measure = []
    per_instance_perfect_cluster_ratio = []

    with torch.no_grad():
        for sample, filepath in tqdm(dataloader, desc="Evaluating test instances"):
            instance_name = os.path.splitext(os.path.basename(filepath))[0]
            sample = sample.to(device)

            c_feat_t = sample['constraint'].x.float()
            v_feat_t = sample['variable'].x.float()
            e_ind_t = sample['constraint', 'includes', 'variable'].edge_index
            e_feat_t = sample['constraint', 'includes', 'variable'].edge_attr.float()
            v_labels_t = sample['variable'].y
            var_names = sample['variable'].var_names

            n_cs = torch.tensor([c_feat_t.shape[0]], dtype=torch.int32, device=device)
            n_vs = torch.tensor([v_feat_t.shape[0]], dtype=torch.int32, device=device)
            model_input = (c_feat_t, e_ind_t, e_feat_t, v_feat_t, n_cs, n_vs)
            variable_embeddings = model(model_input, v_labels=None)

            # Filter for foreground variables (those with labels > 0)
            foreground_mask = v_labels_t > 0
            if not torch.any(foreground_mask):
                continue

            fg_embeddings_np = variable_embeddings[foreground_mask].cpu().numpy()
            fg_labels_np = v_labels_t[foreground_mask].cpu().numpy()
            # Get original indices of foreground variables to map back to names
            fg_indices = torch.where(foreground_mask)[0].cpu().numpy()

            all_embeddings_list.append(fg_embeddings_np)
            all_labels_list.append(fg_labels_np)

            # --- DBSCAN Per-instance metric calculation ---
            min_samples = config.EVAL_PARAMS['dbscan_min_samples']
            if len(fg_embeddings_np) < min_samples:
                continue
            # Use hierarchical DBSCAN
            predicted_labels, eps_values = hierarchical_dbscan(fg_embeddings_np, min_samples)
            # DBSCAN can produce noise points (-1 label), which sklearn metrics handle as a separate cluster.
            per_instance_dbscan_ari.append(adjusted_rand_score(fg_labels_np, predicted_labels))
            per_instance_dbscan_nmi.append(normalized_mutual_info_score(fg_labels_np, predicted_labels))
            per_instance_dbscan_v_measure.append(v_measure_score(fg_labels_np, predicted_labels))

            # --- Calculate Perfect Cluster Recovery Rate ---
            if true_n_clusters > 0:
                # 1. 将真实标签和预测标签都转换为集合的形式
                true_clusters = collections.defaultdict(set)
                for i, label in enumerate(fg_labels_np):
                    true_clusters[label].add(i) # 使用索引作为变量标识

                predicted_clusters = collections.defaultdict(set)
                for i, label in enumerate(predicted_labels):
                    if label != -1: # 忽略噪声点
                        predicted_clusters[label].add(i)

                # 2. 检查每个真实簇是否被完美复现
                perfectly_recovered_count = 0
                # 将真实簇的变量集合转换为frozenset，以便在集合中查找
                true_cluster_sets = {frozenset(v) for v in true_clusters.values()}
                
                for pred_set in predicted_clusters.values():
                    if frozenset(pred_set) in true_cluster_sets:
                        perfectly_recovered_count += 1
                
                recovery_ratio = perfectly_recovered_count / true_n_clusters
                per_instance_perfect_cluster_ratio.append(recovery_ratio)

            # --- Map Clusters to Variable Names ---
            # DBSCAN将噪声点标记为-1，我们将其单独列出
            clusters = collections.defaultdict(list)
            noise_points = []
            for i, cluster_id in enumerate(predicted_labels):
                original_var_index = fg_indices[i]
                var_name = var_names[original_var_index]
                if cluster_id == -1:
                    noise_points.append(var_name)
                else:
                    clusters[cluster_id].append(var_name)

            # --- Prepare and Save Structured Data (JSON) ---
            output_data = {
                "instance_name": instance_name,
                "clustering_parameters": {
                    "algorithm": "hierarchical_dbscan",
                    "min_samples": min_samples,
                    "eps_values_used": eps_values
                },
                "summary": {
                    "num_clusters": len(clusters),
                    "num_noise_points": len(noise_points)
                },
                "clusters": {str(k): sorted(v) for k, v in clusters.items()},
                "noise_points": sorted(noise_points)
            }

            os.makedirs(os.path.join(output_dir, "cluster_outputs"), exist_ok=True)
            output_filepath = os.path.join(output_dir, "cluster_outputs", f"{instance_name}.json")
            with open(output_filepath, 'w') as f:
                json.dump(output_data, f, indent=4)

    if not all_embeddings_list:
        return None

    # --- Aggregate all embeddings for global metrics ---
    all_embeddings = np.concatenate(all_embeddings_list, axis=0)
    all_labels = np.concatenate(all_labels_list, axis=0)

    log("\nCalculating evaluation metrics...", logfile)

    # Average Per-Instance Metrics for DBSCAN
    avg_dbscan_ari = np.mean(per_instance_dbscan_ari) if per_instance_dbscan_ari else 0.0
    avg_dbscan_nmi = np.mean(per_instance_dbscan_nmi) if per_instance_dbscan_nmi else 0.0
    avg_dbscan_v_measure = np.mean(per_instance_dbscan_v_measure) if per_instance_dbscan_v_measure else 0.0
    avg_perfect_cluster_ratio = np.mean(per_instance_perfect_cluster_ratio) if per_instance_perfect_cluster_ratio else 0.0

    # The number of instances evaluated for KMeans and DBSCAN might differ if one algorithm
    # has stricter requirements (e.g., KMeans needs n_samples > n_clusters).
    # We report the count for KMeans as it's typically more restrictive.
    return {
        "num_samples": len(all_labels),
        "true_n_clusters": true_n_clusters,
        "num_instances_evaluated_kmeans": len(per_instance_ari),
        "num_instances_evaluated_dbscan": len(per_instance_dbscan_ari),
        "avg_per_instance_dbscan_ari": avg_dbscan_ari,
        "avg_per_instance_dbscan_nmi": avg_dbscan_nmi,
        "avg_per_instance_dbscan_v_measure": avg_dbscan_v_measure,
        "avg_perfect_cluster_ratio": avg_perfect_cluster_ratio,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a GNN model trained with contrastive learning.")
    parser.add_argument('problem', help='MILP instance type to process.', choices=['facilities', 'osif'])
    args = parser.parse_args()

    model_dir = config.MODELS_DIR / args.problem / "GCNPolicy"
    model_file = model_dir / 'best_params.pkl'
    test_folders = config.TEST_FOLDERS
    output_dir = config.RESULTS_DIR / args.problem / "GCNPolicy"
    os.makedirs(output_dir, exist_ok=True)
    logfile = output_dir / 'test_log.txt' 

    log(f"Problem: {args.problem}", logfile)
    log(f"Loading model from: {model_file}", logfile)
    log(f"Testing on data from: {test_folders}", logfile)

    if not model_file.exists():
        log(f"ERROR: Model file not found at {model_file}", logfile)
        exit(1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}", logfile)

    model = GCNPolicy(emb_size=config.MODEL_PARAMS['emb_size'])
    model.restore_state(model_file)
    model.to(device)
    log("Model loaded successfully.", logfile)

    for folder_name in test_folders:
        full_folder_path = config.PROCESSED_DATA_DIR / args.problem / folder_name
        log(f"\n--- Evaluating folder: {folder_name} ---", logfile)
        if not full_folder_path.exists():
            log(f"ERROR: Test data folder not found: {full_folder_path}", logfile)
            continue

        test_files = sorted([str(p) for p in pathlib.Path(full_folder_path).glob('*.pt')])
        if not test_files:
            log(f"No .pt files found in {full_folder_path}. Skipping.", logfile)
            continue
        
        true_n_clusters = config.TEST_CONFIG.get(folder_name, {}).get('true_n_clusters', 0)
        log(f"Starting evaluation on {len(test_files)} test samples from '{folder_name}' (true #clusters: {true_n_clusters})", logfile)

        test_dataset = MILPDataset(test_files)
        test_loader = DataLoader(test_dataset, batch_size=config.EVAL_PARAMS['test_batch_size'], shuffle=False, num_workers=0, collate_fn=simple_collate)

        metrics = evaluate(model, test_loader, device, true_n_clusters, logfile)

        log(f"\n--- Clustering Evaluation Report for '{folder_name}' ---", logfile)
        if metrics:
            log(f"  Total foreground variables evaluated: {metrics['num_samples']}", logfile)
    
            log("\n--- DBSCAN Per-Instance Metrics (averaged over valid instances) ---", logfile)
            log(f"  Number of instances evaluated for DBSCAN: {metrics['num_instances_evaluated_dbscan']}", logfile)
            log(f"  Average Perfect Cluster Recovery Rate: {metrics['avg_perfect_cluster_ratio']:.4f}", logfile)
            log(f"  Average V-Measure (DBSCAN): {metrics['avg_per_instance_dbscan_v_measure']:.4f}", logfile)
            log(f"  Average Adjusted Rand Index (ARI) (DBSCAN): {metrics['avg_per_instance_dbscan_ari']:.4f}", logfile)
            log(f"  Average Normalized Mutual Information (NMI) (DBSCAN): {metrics['avg_per_instance_dbscan_nmi']:.4f}", logfile)

        else:
            log("  No foreground variables found in the test set to evaluate.", logfile)
        log("--- End of Report ---", logfile)
