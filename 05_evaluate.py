# 测试聚类后的析取结构在实际求解中的速度提升（遇到困难，未完成）
import os
import json
import argparse
import gurobipy as gp
from gurobipy import GRB
import time
import collections
import numpy as np
import csv
import config
from utilities import log
from OSIF_gurobi_callback import init_log_file, log_performance, build_model


def build_cglp_for_facility(model, facility_idx, facility_to_customers, demands):
    """
    为指定的设施构建割平面生成线性规划（CGLP）模型。
    参考自 CFLP_gurobi_callback.py。
    """
    customers_for_i = facility_to_customers.get(facility_idx, [])
    if not customers_for_i:
        return None

    cglp = gp.Model(f"CGLP_{facility_idx}")
    cglp.setParam('OutputFlag', 0)
    cglp.setParam('Method', 0)  # Primal Simplex for CGLP

    # CGLP 变量 (割平面系数)
    alpha = cglp.addVars(customers_for_i, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="alpha")
    beta = cglp.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")
    delta = cglp.addVar(lb=-GRB.INFINITY, name="delta")

    # CGLP 约束 (定义有效割平面的空间)
    # 这些约束来自于与原始问题相关的对偶信息
    for j in customers_for_i:
        # 对应于 x_ij <= y_i 的对偶
        cglp.addConstr(alpha[j] + beta >= 0)
    # 对应于 sum(d_ij * x_ij) <= cap * y_i 的对偶
    cglp.addConstr(gp.quicksum(demands[facility_idx, j] * alpha[j] for j in customers_for_i) + config.INSTANCE_GEN['facilities']['train']['dimension'] * beta >= delta)

    # 归一化约束，防止CGLP无界
    cglp.addConstr(gp.quicksum(alpha[j] for j in customers_for_i) + beta <= 1)
    cglp.addConstr(gp.quicksum(alpha[j] for j in customers_for_i) + beta >= -1)

    return {'model': cglp, 'alpha': alpha, 'beta': beta, 'delta': delta}

def build_cglp_for_osif(model, layer_idx, neuron_idx):
    """
    为指定的 OSIF 神经元构建割平面生成线性规划（CGLP）模型。
    """
    nn_params = model._nn_params
    current_layer_num = layer_idx + 1
    prev_layer_size = nn_params[f'm{current_layer_num}'].shape[1]

    w = nn_params[f'm{current_layer_num}'][neuron_idx]
    b = nn_params[f'b{current_layer_num}'][neuron_idx]

    cglp = gp.Model(f"CGLP_l{layer_idx}_n{neuron_idx}")
    cglp.setParam('OutputFlag', 0)

    # CGLP 决策变量
    delta = cglp.addVar(lb=-GRB.INFINITY, name="delta")
    d_l = cglp.addVars(prev_layer_size, lb=-1.0, ub=1.0, name="d_l")
    d_l_plus_1 = cglp.addVar(lb=-1.0, ub=1.0, name="d_l_plus_1")
    u1 = cglp.addVars(3, lb=0, name="u1")
    u2 = cglp.addVars(3, lb=0, name="u2")

    # CGLP 约束 (使用此神经元固定的 w 和 b)
    cglp.addConstr(delta <= (u1[1] - u1[0]) * b)
    cglp.addConstr(delta <= u2[0] * b)
    cglp.addConstrs((d_l[j] >= (u1[0] - u1[1]) * w[j] for j in range(prev_layer_size)))
    cglp.addConstr(d_l_plus_1 >= u1[1] - u1[0] + u1[2])
    cglp.addConstrs((d_l[j] >= -u2[0] * w[j] for j in range(prev_layer_size)))
    cglp.addConstr(d_l_plus_1 >= u2[1] - u2[2])

    return {
        'model': cglp,
        'vars': {'delta': delta, 'd_l': d_l, 'd_l_plus_1': d_l_plus_1}
    }


def dynamic_cut_callback(model, where):
    """
    Gurobi 回调函数，为机器学习发现并通过验证的析取动态生成割平面。
    """
    if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
        callback_stats = model._callback_stats

        if callback_stats['rootUb'] is None:
            callback_stats['startTime'] = time.time()

        start_time = time.time()
        # 仅在根节点 (node count = 0) 运行以添加初始割平面
        # count = model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT)

        # 在根节点求解后记录根节点信息
        # if count == 0:
        if model._once == False:
            model._once = True
            callback_stats['rootUb'] = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
            callback_stats['rootLb'] = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            callback_stats['rootTime'] = model.cbGet(GRB.Callback.RUNTIME)
            # return
        else:
        # if count > 0:
            callback_stats['time'] += time.time() - start_time
            return
        
            # 从模型对象中检索所需数据
        x_vars = model._vars_x
        nn_params = model._nn_params
        TOL = 1e-6
        
        # 遍历所有已验证的析取（即CGLP模型）
        for (layer_idx, neuron_idx), cglp_info in model._cglp_models.items():
            current_layer_num = layer_idx + 1
            prev_layer_size = nn_params[f'm{current_layer_num}'].shape[1]

            # 获取所需变量的LP松弛解
            x_l_bar = np.array([model.cbGetNodeRel(x_vars[layer_idx][j]) for j in range(prev_layer_size)])
            x_l_plus_1_bar = model.cbGetNodeRel(x_vars[current_layer_num][neuron_idx])

            w = model._nn_params[f'm{current_layer_num}'][neuron_idx]
            b = model._nn_params[f'b{current_layer_num}'][neuron_idx]
            pre_act_val = np.dot(w, x_l_bar) + b

            # 检查ReLU约束是否被违反
            if abs(x_l_plus_1_bar - max(0, pre_act_val)) > TOL:
                cglp = cglp_info['model']
                cglp_vars = cglp_info['vars']
                delta, d_l, d_l_plus_1 = cglp_vars['delta'], cglp_vars['d_l'], cglp_vars['d_l_plus_1']

                # 更新CGLP的目标函数
                obj_expr = delta - sum(d_l[j] * x_l_bar[j] for j in range(prev_layer_size)) - d_l_plus_1 * x_l_plus_1_bar
                cglp.setObjective(obj_expr, GRB.MAXIMIZE)
                cglp.optimize()

                # 如果找到一个有效的割平面
                if cglp.status == GRB.OPTIMAL and cglp.ObjVal > TOL:
                    delta_val = delta.X
                    d_l_val = cglp.getAttr('X', d_l)
                    d_l_plus_1_val = d_l_plus_1.X

                    # 构建割平面表达式
                    cut_expr = gp.quicksum(d_l_val[j] * x_vars[layer_idx][j] for j in range(prev_layer_size)) + \
                                   d_l_plus_1_val * x_vars[current_layer_num][neuron_idx]
                    
                    model.cbLazy(cut_expr >= delta_val)
                    model._callback_stats['nCuts'] += 1


def validate_clusters(model, cluster_data):
    """
    验证GNN预测的聚类是否对应于模型中真实的“和为一”约束。

    Args:
        model (gp.Model): Gurobi模型对象。
        cluster_data (dict): 从JSON文件加载的聚类数据。

    Returns:
        list: 一个包含已验证簇（每个簇是一个变量名集合）的列表。
    """
    # 1. 高效提取模型中所有和为1的约束
    # 我们将真实约束的变量集存储在一个集合中，以便进行快速查找（O(1)平均时间复杂度）
    sum_to_one_constraints = set()
    for constr in model.getConstrs():
        if constr.sense == GRB.EQUAL and constr.rhs == 1.0:
            expr = model.getRow(constr)
            # 确保所有系数都为1且约束非空
            if expr.size() > 0 and all(expr.getCoeff(i) == 1.0 for i in range(expr.size())):
                vars_in_constr = frozenset(expr.getVar(i).VarName for i in range(expr.size()))
                sum_to_one_constraints.add(vars_in_constr)

    # 2. 检查每个预测的聚类是否匹配一个真实的“和为一”约束
    validated_clusters = []
    predicted_clusters = [set(v) for v in cluster_data['clusters'].values()]
    
    for pred_set in predicted_clusters:
        # 使用 frozenset 是因为 set 对象本身是不可哈希的，不能作为集合的元素
        if frozenset(pred_set) in sum_to_one_constraints:
            validated_clusters.append(pred_set)
            
    return validated_clusters

def main(problem, test_folder):
    mps_dir = config.MPS_DATA_DIR / problem / test_folder
    cluster_dir = config.RESULTS_DIR / problem / "GCNPolicy"
    log_dir = config.RESULTS_DIR / problem / "GCNPolicy_eval"
    os.makedirs(log_dir, exist_ok=True)
    logfile = log_dir / f"eval_log_{test_folder}.txt"
    csv_path = log_dir / f"comparison_{test_folder}.csv"

    log(f"--- Starting Evaluation for '{test_folder}' ---", logfile)
    log(f"Reading MPS files from: {mps_dir}", logfile)
    log(f"Reading cluster JSONs from: {cluster_dir}", logfile)
    log(f"Saving comparison results to: {csv_path}", logfile)

    # --- 1. 初始化CSV文件并写入表头 ---
    header = [
        "instance_name",
        "baseline_time", "baseline_obj", "baseline_gap", "baseline_nodes",
        "callback_time", "callback_obj", "callback_gap", "callback_nodes",
        "nCuts", "rootUb", "rootLb", "rootTime", "validated_disjunctions"
    ]
    if not os.path.exists(csv_path):
        init_log_file(csv_path, header)

    # --- 2. 遍历所有实例 ---
    json_files = sorted(list((cluster_dir / "cluster_outputs").glob(f"*.json")))

    count = 0
    for json_path in json_files:
        if count > 1:
            break
        count += 1

        instance_name = json_path.stem
        raw_path = config.RAW_DATA_DIR / problem / test_folder / f"{instance_name}.npz"
        mps_path = mps_dir / f"{instance_name}.mps"

        if not mps_path.exists():
            log(f"Skipping {instance_name}: MPS file not found.", logfile)
            continue

        results_row = {"instance_name": instance_name}
        log(f"\n--- Processing instance: {instance_name} ---", logfile)

        # --- 3. 基准求解 (不使用回调) ---
        log(f"  - Solving baseline...", logfile)
        baseline_model = gp.read(str(mps_path))
        # baseline_model.setParam('OutputFlag', 0)
        baseline_model.setParam('TimeLimit', 300)

        start_time = time.time()
        baseline_model.optimize()
        solve_time = time.time() - start_time

        results_row['baseline_time'] = solve_time
        results_row['baseline_obj'] = baseline_model.ObjVal if baseline_model.SolCount > 0 else float('-inf')
        results_row['baseline_gap'] = baseline_model.MIPGap * 100 if hasattr(baseline_model, 'MIPGap') else 100.0
        results_row['baseline_nodes'] = baseline_model.NodeCount
        log(f"    Baseline finished in {solve_time:.2f}s. "
            f"Status: {baseline_model.status}. "
            f"Objective: {results_row['baseline_obj']}. "
            f"MIPGap: {results_row['baseline_gap']}. "
            f"Nodes: {results_row['baseline_nodes']}", logfile)


        # --- 4. 回调求解 ---
        log(f"  - Solving with callback...", logfile)
        # 重新读取模型以保证求解环境干净
        model = build_model(f"{instance_name}.npz", 0, 10.0)
        
        model.setParam('TimeLimit', 1800)
        model.setParam('LazyConstraints', 1)

        # 4.1 验证聚类结果的准确性
        with open(json_path, 'r') as f:
            cluster_data = json.load(f)

        # 调用新的验证函数
        validated_clusters = validate_clusters(model, cluster_data)
        
        log(f"    Found {len(cluster_data['clusters'])} predicted clusters. "
            f"Validated {len(validated_clusters)} as correct disjunctions.", logfile)
        
        results_row['validated_disjunctions'] = len(validated_clusters)

        # 4.2 为通过验证的析取添加CGLP
        model._cglp_models = {}
        model._nn_params = np.load(raw_path)

        for cluster in validated_clusters:
            layer_idx, neuron_idx = -1, -1
            for var_name in cluster:
                if var_name.startswith('ind_disjunction_'):
                    try:
                        # e.g., ind_disjunction_5_disjunct_2_l0n10
                        ln_part = var_name.split('_')[-1] # 'l0n10'
                        l_pos = ln_part.find('l')
                        n_pos = ln_part.find('n')
                        layer_idx = int(ln_part[l_pos+1:n_pos])
                        neuron_idx = int(ln_part[n_pos+1:])
                        break
                    except (ValueError, IndexError):
                        continue
            
            if layer_idx != -1 and neuron_idx != -1:
                cglp_model_data = build_cglp_for_osif(model, layer_idx, neuron_idx)
                if cglp_model_data:
                    model._cglp_models[(layer_idx, neuron_idx)] = cglp_model_data

        if not model._cglp_models:
            log("    No valid disjunctions found to apply cuts. Skipping solve with callback.", logfile)
            results_row.update({
                'callback_time': 0, 'callback_obj': 'N/A', 'callback_gap': 'N/A',
                'callback_nodes': 'N/A', 'nCuts': 0, 'rootUb': None, 'rootLb': None, 'rootTime': None
            })
            log_performance(csv_path, results_row, float_precision=4)
            continue

        # 4.3 初始化指标并使用回调求解
        callback_stats = {
            'startTime': None,
            'time': 0, 
            'nCuts': 0, 
            'rootUb': None, 
            'rootLb': None, 
            'rootTime': None
        }
        model._callback_stats = callback_stats
        model._once = False

        # 使用回调函数求解
        log(f"    Solving with dynamic cuts for {len(model._cglp_models)} disjunctions...", logfile)
        start_time = time.time()
        model.optimize(dynamic_cut_callback)
        solve_time = time.time() - start_time

        results_row['callback_time'] = solve_time
        results_row['callback_obj'] = model.ObjVal if model.SolCount > 0 else float('-inf')
        results_row['callback_gap'] = model.MIPGap * 100 if hasattr(model, 'MIPGap') else 100.0
        results_row['callback_nodes'] = model.NodeCount
        results_row.update(callback_stats)
        
        log_performance(csv_path, results_row, float_precision=4)
        log(f"    Callback finished in {solve_time:.2f}s. Objective: {results_row['callback_obj']}. Cuts: {results_row['nCuts']}", logfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate GNN-discovered disjunctions by applying cuts in Gurobi.")
    parser.add_argument('problem', help='MILP instance type to process.', choices=['facilities', 'osif'])
    args = parser.parse_args()

    test_folder = config.EVAL_FOLDERS[0]
    main(args.problem, test_folder)
