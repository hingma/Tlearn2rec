# 在多个文件中调用的函数

import datetime
import numpy as np
import scipy.sparse as sp
import pyscipopt as scip
import argparse
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import gurobipy as gp
from gurobipy import GRB


def valid_seed(seed):
    """Check whether seed is a valid random seed or not."""
    seed = int(seed)
    if seed < 0 or seed > 2**32 - 1:
        raise argparse.ArgumentTypeError(
                "seed must be any integer between 0 and 2**32 - 1 inclusive")
    return seed

def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)

def hierarchical_dbscan(embeddings, min_samples, start_scale=0.2, step_scale=0.2, max_scale=4.0):
    """
    Performs hierarchical DBSCAN by running it iteratively with increasing eps until no noise points are left.
    It starts with a scale factor `start_scale` and increases it by `step_scale` in each iteration.
    Points clustered in one round are excluded from subsequent rounds. The process stops when all points
    are clustered or `max_scale` is reached.
    """
    base_eps = find_dbscan_eps(embeddings, min_samples, scaling_factor=1.0) # Get the raw knee point
    if base_eps == 0:
        base_eps = 0.1
 
    remaining_indices = np.arange(len(embeddings))
    final_labels = np.full(len(embeddings), -1, dtype=int)
    cluster_offset = 0
    eps_values = []
 
    scale_factor = start_scale
    while len(remaining_indices) > 0 and scale_factor <= max_scale:
        eps = base_eps * scale_factor
        eps_values.append(eps)
 
        if len(remaining_indices) < min_samples:
            break
 
        current_embeddings = embeddings[remaining_indices]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        predicted_labels = dbscan.fit_predict(current_embeddings)
 
        clustered_mask = predicted_labels != -1
        if np.any(clustered_mask):
            newly_clustered_original_indices = remaining_indices[clustered_mask]
            new_labels = predicted_labels[clustered_mask] + cluster_offset
            final_labels[newly_clustered_original_indices] = new_labels
 
            cluster_offset = new_labels.max() + 1
            remaining_indices = remaining_indices[~clustered_mask]
 
        scale_factor += step_scale
 
    return final_labels, eps_values

def find_dbscan_eps(embeddings, min_samples, scaling_factor = 1.0):
    """
    使用k-distance图的方法自动为DBSCAN寻找一个合适的eps值。
    它寻找距离图中斜率变化最剧烈的“拐点”。
    """
    # 1. 计算每个点到其k-th最近邻的距离 (k = min_samples - 1)
    k = min_samples - 1
    if k <= 0: # Handle case where min_samples is 1 or less, though for DBSCAN min_samples >= 2 is typical
        return 0.1 * scaling_factor # A small default eps if k is not meaningful for distance calculation

    neighbors = NearestNeighbors(n_neighbors=k + 1)
    neighbors_fit = neighbors.fit(embeddings)
    distances, _ = neighbors_fit.kneighbors(embeddings)
    
    # 获取到第k个邻居的距离并排序
    k_distances = np.sort(distances[:, k], axis=0)
    
    # 2. 找到“拐点”
    # "拐点"是曲线上离连接首末两点的直线最远的点。
    n_points = len(k_distances)
    all_coords = np.vstack((range(n_points), k_distances)).T
    
    first_point = all_coords[0]
    last_point = all_coords[-1]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    
    # 找到最大距离对应的点，其y坐标（即k-distance）就是最佳eps
    best_eps_index = np.argmax(dist_to_line)
    best_eps = k_distances[best_eps_index]
    
    # 启发式方法找到的eps有时会偏大，导致簇被合并。
    # 乘以一个小于1的因子可以收紧邻域，帮助分离靠得近的簇。
    # 0.75是一个经验值，可以根据实验效果调整。
    return best_eps * scaling_factor

def CFLP_seperating_disjunctive_callback_once(model, where):
    """
    Gurobi 回调函数，用于在MIP节点上添加 seperating disjunctive cuts。
    该函数通过求解一个割平面生成线性规划（CGLP）来找到最深的有效不等式。
    此版本仅在根节点运行。
    """
    # 仅在根节点 (Node 0) 找到LP最优解时运行

    if where == GRB.Callback.MIPNODE and \
       model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL:
        
        if model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT) > 0:
            return
        # 记录初始根节点界 (只记录一次)
        if model._initial_root_bound is None and model._once:
            model._initial_root_bound = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBND)

        if model._once:
            return
        else:
            model._once = True
            

        y_vals = model.cbGetNodeRel(model._vars_y)
        x_vals = model.cbGetNodeRel(model._vars_x)

        facilities = model._facilities


            # print(f"Callback: 初始根节点界: {model._initial_root_bound:.4f}")

        # 遍历所有设施，为y_i为分数的设施寻找割平面
        for i in facilities:
            if 0.001 < y_vals[i] < 0.999:
                cglp_data = model._cglp_models.get(i)

                cglp = cglp_data['model']
                alpha = cglp_data['alpha']
                beta = cglp_data['beta']
                delta = cglp_data['delta']
                customers_for_i = list(alpha.keys())

                # CGLP 目标函数: max (beta - alpha * z_bar)
                cglp.setObjective(
                    (delta - gp.quicksum(alpha[j] * x_vals[i, j] for j in customers_for_i) - beta * y_vals[i]),
                    GRB.MAXIMIZE
                )

                # 求解CGLP
                cglp.reset()
                cglp.optimize()

                # 如果找到一个被违犯的割平面 (目标值 > 0)
                if cglp.Status == GRB.OPTIMAL and cglp.ObjVal > 0.0:
                    # 获取割平面系数
                    alpha_x_sol = cglp.getAttr('X', alpha)
                    alpha_y_sol = beta.X
                    beta_sol = delta.X

                    # 添加割平面到主模型
                    cut_lhs = gp.quicksum(alpha_x_sol[j] * model._vars_x[i, j] for j in customers_for_i) + \
                              alpha_y_sol * model._vars_y[i]
                    
                    model.cbLazy(cut_lhs >= beta_sol)
                    model._cuts_added += 1

                    print(f"Callback: 在根节点添加了析取割平面，最大违反量为 {cglp.ObjVal}。")


def CFLP_seperating_disjunctive_callback_root(model, where):
    """
    Gurobi 回调函数，用于在MIP节点上添加 seperating disjunctive cuts。
    该函数通过求解一个割平面生成线性规划（CGLP）来找到最深的有效不等式。
    此版本仅在根节点运行。
    """
    # 仅在根节点 (Node 0) 找到LP最优解时运行
    if where == GRB.Callback.MIPNODE and \
       model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL:
        
        start_at_node = 1

        count = model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT)
        if count > start_at_node + 1:
            return
        # 记录初始根节点界 (只记录一次)
        if model._initial_root_bound is None and count > start_at_node:
            model._initial_root_bound = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBND)

        if count > start_at_node or count < start_at_node:
            return
        
        y_vals = model.cbGetNodeRel(model._vars_y)
        x_vals = model.cbGetNodeRel(model._vars_x)

        demands = model._demands
        capacity = model._capacity
        facilities = model._facilities
        violated_cuts = []

        for i in facilities:
            if 0.001 < y_vals[i] < 0.999:
                cglp_data = model._cglp_models.get(i)
                customers_for_i = list(cglp_data['alpha'].keys())

                # --- 更优的判断条件 ---
                # 只有当LP松弛解实际违反了凸包约束时，才尝试生成割平面
                capacity_violated = sum(demands[i, j] * x_vals[i, j] for j in customers_for_i) > capacity * y_vals[i] + 1e-6
                assignment_violated = sum(x_vals[i, j] for j in customers_for_i) > min(sum(1 for _ in customers_for_i), capacity) * y_vals[i] + 1e-6

                if not (capacity_violated or assignment_violated):
                    continue # 如果没有违反，跳过此设施，避免不必要的CGLP求解

                if not cglp_data:
                    continue

                cglp = cglp_data['model']
                alpha = cglp_data['alpha']
                beta = cglp_data['beta']
                delta = cglp_data['delta']

                # CGLP 目标函数: max (beta - alpha * z_bar)
                cglp.setObjective(
                    (delta - gp.quicksum(alpha[j] * x_vals[i, j] for j in customers_for_i) - beta * y_vals[i]),
                    GRB.MAXIMIZE
                )

                # 求解CGLP
                cglp.reset()
                cglp.optimize()

                # 如果找到一个被违犯的割平面 (目标值 > 0)
                print(cglp.ObjVal)
                if cglp.ObjVal > 0.0:
                    cut_info = {
                        'violation': cglp.ObjVal,
                        'facility': i,
                        'lhs': gp.quicksum(cglp.getAttr('X', alpha)[j] * model._vars_x[i, j] for j in customers_for_i) + \
                                beta.X * model._vars_y[i],
                        'rhs': delta.X
                    }
                    violated_cuts.append(cut_info)

        # 2. 找出违反量最大的前15个割平面
        violated_cuts.sort(key=lambda x: x['violation'], reverse=True)
        cuts_to_add_count = 15
        top_cuts = violated_cuts[:cuts_to_add_count]

        # 3. 按照设施索引的原始顺序重新排序，以保证添加顺序的确定性
        top_cuts.sort(key=lambda x: x['facility'])

        # 4. 添加这些割平面
        for cut_info in top_cuts:
            model.cbLazy(cut_info['lhs'] >= cut_info['rhs'])
            model._cuts_added += 1



def CFLP_control_callback(model, where):
    """
    对照组，只记录指标
    """
    # 仅在根节点 (Node 0) 找到LP最优解时运行
    if where == GRB.Callback.MIPNODE and \
       model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL:

        start_at_node = 1

        count = model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT)
        if count > start_at_node + 1:
            return
        # 记录初始根节点界 (只记录一次)
        if model._initial_root_bound is None and count > start_at_node:
            model._initial_root_bound = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBND)

        if count > start_at_node or count < start_at_node:
            return


            # print(f"Callback: 初始根节点界: {model._initial_root_bound:.4f}")
