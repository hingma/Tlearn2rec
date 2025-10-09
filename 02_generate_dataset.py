# 将原始数据转化为solver可直接求解的mps格式，储存在data/mps中
# 同时建立数据对应的二分图，并添加训练所需要的标签，将完整的图结构保存为.pt格式，储存在data/processed
# 使用方法：python 02_generate_dataset.py [problem]
import os
import argparse
import glob
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from torch_geometric.data import HeteroData
import multiprocessing
from functools import partial
import scipy.sparse
from scipy.sparse.csgraph import shortest_path
import shutil
import networkx as nx
import config

# Assuming CFLP.py is in the same directory or in python path
from CFLP import DataLoader, build_gurobi_model
from OSIF_gurobi_callback import build_model

def CFLP_from_raw_to_mps(raw_file, mps_dir):
    """
    Converts a single raw instance file (.txt) to MPS format using Gurobi.
    """
    try:   
        # 1. Build model
        data_loader = DataLoader(raw_file)
        model = build_gurobi_model(data_loader)

        # 2. Save to .mps file
        instance_name = os.path.splitext(os.path.basename(raw_file))[0]
        mps_path = os.path.join(mps_dir, f"{instance_name}.mps")
        model.write(mps_path)
        print(f"    Successfully saved to {mps_path}")
        return model

    except Exception as e:
        print(f"    Failed to process {raw_file}: {e}")

def OSIF_from_raw_to_mps(raw_file, mps_dir, target_class, epsilon):
    try:   
        model = build_model(raw_file, target_class, epsilon)
        instance_name = os.path.splitext(os.path.basename(raw_file))[0]
        mps_path = os.path.join(mps_dir, f"{instance_name}.mps")
        model.write(mps_path)
        print(f"    Successfully saved to {mps_path}")
        return model

    except Exception as e:
        print(f"    Failed to process {raw_file}: {e}")

def calculate_features(model, n_anchors):
    """
    Calculates node and edge features for a Gurobi model.
    Features are primarily based on graph structure to be robust to coefficient changes.
    This includes anchor-based features to give nodes a sense of "position" in the graph.

    Returns:
        tuple: A tuple containing:
            - final_constr_features (np.ndarray): Features for each constraint.
            - final_var_features (np.ndarray): Features for each variable.
            - edge_features_list (dict): Features for each edge.
    """
    model.update()
    constrs = model.getConstrs()
    variables = model.getVars()
    n_constraints = len(constrs)
    n_variables = len(variables)
    rng = np.random.default_rng(seed=42)  # For reproducible anchor selection

    # 1. Build adjacency matrix for the whole graph (constraints + variables)
    var_map = {v: i for i, v in enumerate(variables)}
    # 存储邻接矩阵中值为1的元素的坐标
    rows, cols = [], []
    for i, constr in enumerate(constrs):
        expr = model.getRow(constr)
        for j in range(expr.size()):
            var = expr.getVar(j)
            var_idx = var_map.get(var)
            if var_idx is not None:
                # Node indices: constraints: 0..n_constraints-1; variables: n_constraints..n_constraints+n_variables-1
                rows.append(i) # 约束索引
                cols.append(n_constraints + var_idx) # 变量索引

    n_total_nodes = n_constraints + n_variables 
    # np.ones(len(rows)): 在所有记录的 (row, col) 位置上都填入 1，表示有边存在
    adj = scipy.sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_total_nodes, n_total_nodes)
    ).tocsr() # CSR（Compressed Sparse Row）格式
    adj = adj + adj.T  # 添加从变量到约束的边，直接转置相加即可

    # 2. Centrality Feature Calculation (Optimized)
    k_betweenness = 32  # 使用256个采样节点进行近似计算
    if n_total_nodes > k_betweenness:
        G = nx.from_scipy_sparse_array(adj)
        betweenness_centrality = nx.betweenness_centrality(G, k=k_betweenness, normalized=True, seed=42)
    else:
        # 对于小图，仍然可以进行精确计算
        G = nx.from_scipy_sparse_array(adj)
        betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

    # 分离约束和变量的中心性值
    constr_betweenness = np.array([betweenness_centrality.get(i, 0.0) for i in range(n_constraints)], dtype=np.float32).reshape(-1, 1)
    var_betweenness = np.array([betweenness_centrality.get(i + n_constraints, 0.0) for i in range(n_variables)], dtype=np.float32).reshape(-1, 1)

    # 3. Anchor-based Feature Calculation
    
    # 使用 Dijkstra 最短距离
    # dist_matrix = shortest_path(
    #     csgraph=adj,
    #     method='auto',  # Use Breadth-First Search for unweighted graphs
    #     directed=False,
    #     indices=anchor_nodes,
    #     unweighted=True # 我们的图是无权的
    # )
    # # Handle disconnected components (inf distance)
    # max_dist = np.max(dist_matrix[np.isfinite(dist_matrix)], initial=0)
    # dist_matrix[np.isinf(dist_matrix)] = max_dist + 1
    
    # # Normalize distances
    # if max_dist > 0:
    #     dist_matrix /= (max_dist + 1)

    # 使用 Adamic-Adar 方法计算距离
    anchor_nodes = rng.choice(n_total_nodes, size=min(n_anchors, n_total_nodes), replace=False)
    
    degrees = np.array(adj.sum(axis=1)).flatten()
    log_degrees = np.log(degrees + 1) 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.reciprocal(log_degrees)
    weights[np.isinf(weights)] = 0 # Handle division by zero for log(1)

    adj_weighted = adj.multiply(weights).tocsr()

    similarity_matrix = (adj[anchor_nodes, :] @ adj_weighted.T).toarray()
    
    dist_matrix = 1.0 / (1.0 + similarity_matrix)
    
    constr_anchor_features = dist_matrix[:, :n_constraints].T
    var_anchor_features = dist_matrix[:, n_constraints:].T

    # Pad with zeros if the graph is smaller than n_anchors
    if constr_anchor_features.shape[1] < n_anchors:
        padding = np.zeros((constr_anchor_features.shape[0], n_anchors - constr_anchor_features.shape[1]), dtype=np.float32)
        constr_anchor_features = np.hstack([constr_anchor_features, padding])
    if var_anchor_features.shape[1] < n_anchors:
        padding = np.zeros((var_anchor_features.shape[0], n_anchors - var_anchor_features.shape[1]), dtype=np.float32)
        var_anchor_features = np.hstack([var_anchor_features, padding])

    # 4. Calculate other structural features (non-coefficient based)
    var_feature_names = ["is_binary", "is_integer", "is_continuous", "degree", "is_in_sum2one"] # 'betweenness' will be added
    constr_feature_names = ["is_le", "is_ge", "is_eq", "degree", "num_binary_vars", "ratio_binary_vars"] # 'betweenness' will be added
    
    var_features_np = np.zeros((n_variables, len(var_feature_names)), dtype=np.float32) 
    constr_features_np = np.zeros((n_constraints, len(constr_feature_names)), dtype=np.float32) 

    sum_to_one_vars = set()
    for constr in constrs:
        if constr.sense == GRB.EQUAL and constr.rhs == 1.0:
            expr = model.getRow(constr)
            is_sum_to_one = expr.size() > 0 and all(expr.getCoeff(i) == 1.0 for i in range(expr.size()))
            if is_sum_to_one:
                for i in range(expr.size()):
                    sum_to_one_vars.add(expr.getVar(i).VarName)

    # Constraint Features
    for i, constr in enumerate(constrs):
        constr_features_np[i, constr_feature_names.index("is_le")] = 1 if constr.sense == GRB.LESS_EQUAL else 0
        constr_features_np[i, constr_feature_names.index("is_ge")] = 1 if constr.sense == GRB.GREATER_EQUAL else 0
        constr_features_np[i, constr_feature_names.index("is_eq")] = 1 if constr.sense == GRB.EQUAL else 0
        
        expr = model.getRow(constr)
        num_vars_in_constr = expr.size()
        
        if num_vars_in_constr > 0:
            constr_features_np[i, constr_feature_names.index("degree")] = num_vars_in_constr
            num_binary_vars = sum(1 for j in range(num_vars_in_constr) if expr.getVar(j).vtype == GRB.BINARY)
            constr_features_np[i, constr_feature_names.index("num_binary_vars")] = num_binary_vars
            constr_features_np[i, constr_feature_names.index("ratio_binary_vars")] = num_binary_vars / num_vars_in_constr

    # Variable Features
    for i, var in enumerate(variables):
        var_features_np[i, var_feature_names.index("is_binary")] = 1 if var.vtype == GRB.BINARY else 0
        var_features_np[i, var_feature_names.index("is_integer")] = 1 if var.vtype == GRB.INTEGER else 0
        var_features_np[i, var_feature_names.index("is_continuous")] = 1 if var.vtype == GRB.CONTINUOUS else 0
        var_features_np[i, var_feature_names.index("degree")] = model.getCol(var).size()
        var_features_np[i, var_feature_names.index("is_in_sum2one")] = 1 if var.VarName in sum_to_one_vars else 0

    # 5. Combine all features: structural, centrality, and anchor-based
    final_constr_features = np.hstack([constr_features_np, constr_betweenness, constr_anchor_features]).astype(np.float32)
    final_var_features = np.hstack([var_features_np, var_betweenness, var_anchor_features]).astype(np.float32)

    # 6. Edge features (a single constant feature to indicate existence)
    edge_features_list = {}
    var_map_by_name = {v.VarName: i for i, v in enumerate(variables)}
    for i, constr in enumerate(constrs):
        expr = model.getRow(constr)
        for j in range(expr.size()):
            var = expr.getVar(j)
            var_idx = var_map_by_name.get(var.VarName)
            if var_idx is not None:
                edge_features_list[(i, var_idx)] = [1.0]

    return final_constr_features, final_var_features, edge_features_list

def generate_labels(model):
    """
    为变量生成标签，用于后续的监督式对比学习任务。

    标签策略如下:
    - 属于同一个析取（disjunction）的指示变量（indicator variables）会被赋予相同的整数标签。
      例如，名为 'ind_disjunction_5_disjunct_1' 和 'ind_disjunction_5_disjunct_2' 的变量
      都会被赋予标签 5。这些变量在对比学习中被视为正样本对。
    - 所有其他变量被赋予标签 0，作为“背景”变量，在训练中通常被忽略。

    函数还会生成一个掩码（mask），用于指示哪些变量可用于训练（在此实现中为所有变量）。

    Returns:
        labels (np.ndarray): 一个整数数组，其中每个元素是对应变量的类别标签。
        mask (np.ndarray): 一个布尔数组，在此实现中所有值都为 True。
    """
    # 获取变量并初始化标签
    variables = model.getVars()
    n_variables = len(variables)
    labels = np.zeros(n_variables, dtype=np.int32)
    mask = np.ones(n_variables, dtype=np.bool_) # Train on all variables

    for i, var in enumerate(variables):
        var_name = var.VarName
        if var_name.startswith("ind_disjunction_"):
            parts = var_name.split('_')
            # 预期格式: ind_disjunction_{i}_disjunct_{j}
            if parts[1] == 'disjunction':
                try:
                    disjunction_index = int(parts[2])
                    labels[i] = disjunction_index
                except (ValueError, IndexError):
                    pass  # 如果解析失败，标签保持为0
        # 其他变量默认为背景标签0

    return labels, mask

def _process_single_file(raw_file, mps_dir, processed_dir, problem):
    """
    Processes a single raw instance file:
    1. Converts to MPS
    2. Extracts features and labels
    3. Saves as a PyTorch Geometric Data object.
    """
    print(f"  Processing {os.path.basename(raw_file)}...")
    try:
        if problem == 'facilities':
            model = CFLP_from_raw_to_mps(raw_file, mps_dir)
        elif problem == 'osif':

            model = OSIF_from_raw_to_mps(raw_file, mps_dir, 0, config.OSIF_EPSILON)
        else:
            raise ValueError(f"Unknown problem type: {problem}")

        # Get variable names to store them for later evaluation
        variables = model.getVars()
        var_names = [v.VarName for v in variables]

        # Calculate features matrix
        constraint_features, variable_features, edge_attr = calculate_features(model, n_anchors=config.FEATURE_EXTRACTION['n_anchors'])
        # c. Generate labels (placeholder)
        labels, mask = generate_labels(model)

        # 确保标签和特征的维度匹配
        assert variable_features.shape[0] == labels.shape[0], "特征和标签的数量不匹配！"

        # d. Create HeteroData object
        data = HeteroData()

        # e. Populate node, edge, and label data
        # Node data
        data['constraint'].x = torch.from_numpy(constraint_features)
        data['variable'].x = torch.from_numpy(variable_features)
        data['variable'].var_names = var_names  # Store original variable names

        # Edge data for 'constraint' -> 'variable'
        edge_index_list = [[i, j] for i, j in edge_attr.keys()]
        edge_index_tensor = torch.tensor(list(edge_index_list)).T
        edge_attr_tensor = torch.tensor(list(edge_attr.values()))
        data['constraint', 'includes', 'variable'].edge_index = edge_index_tensor
        data['constraint', 'includes', 'variable'].edge_attr = edge_attr_tensor

        # Add reverse edges to make the graph undirected
        data['variable', 'in', 'constraint'].edge_index = edge_index_tensor.flip([0])
        data['variable', 'in', 'constraint'].edge_attr = edge_attr_tensor

        # Label data
        data['variable'].y = torch.from_numpy(labels)
        data['variable'].train_mask = torch.from_numpy(mask)

        # f. Save Data object
        instance_name = os.path.splitext(os.path.basename(raw_file))[0]
        pt_path = os.path.join(processed_dir, f"{instance_name}.pt")
        torch.save(data, pt_path)
        print(f"    Successfully saved to {pt_path}")
    except Exception as e:
        print(f"    Failed to process {raw_file}: {e}")

def run_preprocessing(problem, instance_dir, mps_dir, processed_dir):
    """
    Processes raw instance files into PyTorch Geometric Data objects in parallel.
    """
    print(f"Processing Raw files from '{instance_dir}' to MPS Data in '{mps_dir}' and PyG Data in '{processed_dir}'...")
    
    os.makedirs(mps_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    if problem == 'facilities':   
        raw_files = glob.glob(os.path.join(instance_dir, '*.txt'))
    elif problem == 'osif':
        raw_files = glob.glob(os.path.join(instance_dir, '*.npz'))
    

    # Use multiprocessing to process files in parallel
    num_processes = multiprocessing.cpu_count()
    print(f"Starting parallel processing with {num_processes} processes...")

    worker_fn = partial(_process_single_file, mps_dir=mps_dir, processed_dir=processed_dir, problem=problem)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(worker_fn, raw_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate graph dataset from MILP instances.")
    parser.add_argument('problem', help='MILP instance type to process.', choices=['facilities', 'osif'])
    args = parser.parse_args()

    instance_dir = str(config.RAW_DATA_DIR)
    mps_dir = str(config.MPS_DATA_DIR)
    processed_dir = str(config.PROCESSED_DATA_DIR)

    instance_subdirs = glob.glob(os.path.join(instance_dir, args.problem, '*'))
    for subdir in instance_subdirs:
        if not os.path.isdir(subdir):
            continue
        if 'test' in subdir:
            continue

        subdir_name = os.path.basename(subdir)
        print(f"\nProcessing instance set: {subdir_name}")
        current_instance_dir = os.path.join(instance_dir, args.problem, subdir_name)
        current_mps_dir = os.path.join(mps_dir, args.problem, subdir_name)
        current_processed_dir = os.path.join(processed_dir, args.problem, subdir_name)

        run_preprocessing(args.problem, current_instance_dir, current_mps_dir, current_processed_dir)

    print("\nDataset generation complete.")