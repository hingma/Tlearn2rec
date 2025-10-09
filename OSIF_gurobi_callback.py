"""
This file implements the Optimal Sparse Input Features problem using a standard
Big-M formulation for the ReLU activation functions.
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
import os
import time
import csv
import math

# 初始化日志文件（如果不存在）
def init_log_file(log_file, header):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # [
        # "problem", "instance", "epsilon", "target", "algoName", "feasible", ...
        # ]
        writer.writerow(header)

# 记录单次运行结果
def log_performance(log_file, result_dict, float_precision=4):
    """
    记录单次运行结果，并将浮点数格式化为指定精度。
    """
    try:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = []
            for value in result_dict.values():
                # 检查值是否为浮点数
                if isinstance(value, float):
                    # 处理特殊的浮点数值，如 inf, -inf, nan
                    if math.isinf(value) or math.isnan(value):
                        row.append(str(value))
                    else:
                        # 格式化为指定精度
                        row.append(f"{value:.{float_precision}f}")
                elif value is None:
                    row.append('') # 对 None 值写入空字符串
                else:
                    row.append(value) # 其他类型（如 int, str）直接添加
            writer.writerow(row)
    except Exception as e:
        print(f"写入文件失败: {e}")


def seperating_disjunctive_callback(model, where):
    """
    Gurobi 回调函数，用于在MIP节点上添加 seperating disjunctive cuts。
    该函数通过求解一个割平面生成线性规划（CGLP）来找到最深的有效不等式。
    """
    # 仅在MIP树的节点处，当LP松弛被最优求解时运行
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
        n_layers = model._nlayers
        cglp_data = model._cglp_data
        TOL = 1e-6

        # 遍历所有隐藏层 (最后一层是线性的，没有ReLU)
        for layer_idx in range(n_layers - 1):
            current_layer_num = layer_idx + 1
            prev_layer_size = nn_params[f'm{current_layer_num}'].shape[1]
            current_layer_size = nn_params[f'm{current_layer_num}'].shape[0]

            # 获取前一层输出 x_l 的LP松弛解 (x_l_bar)，此层所有神经元共享
            x_l_bar = np.array([model.cbGetNodeRel(x_vars[layer_idx][j]) for j in range(prev_layer_size)])

            # 遍历当前层中的每个神经元
            for neuron_idx in range(current_layer_size):
                # 获取当前神经元输出 x_{l+1} 的LP松弛解 (x_l_plus_1_bar)
                x_l_plus_1_bar = model.cbGetNodeRel(x_vars[current_layer_num][neuron_idx])

                # 获取权重 w 和偏置 b 以检查约束是否被违反
                w = nn_params[f'm{current_layer_num}'][neuron_idx]
                b = nn_params[f'b{current_layer_num}'][neuron_idx]

                # 根据松弛解计算预激活值
                pre_act_val = np.dot(w, x_l_bar) + b

                # 检查ReLU约束是否被违反
                if abs(x_l_plus_1_bar - max(0, pre_act_val)) > TOL:
                    # 约束被违反，获取为此神经元预构建的CGLP模型
                    cglp_info = cglp_data[layer_idx][neuron_idx]
                    cglp = cglp_info['model']
                    cglp_vars = cglp_info['vars']
                    delta, d_l, d_l_plus_1 = cglp_vars['delta'], cglp_vars['d_l'], cglp_vars['d_l_plus_1']

                    # 仅更新CGLP的目标函数
                    obj_expr = delta - sum(d_l[j] * x_l_bar[j] for j in range(prev_layer_size)) - d_l_plus_1 * x_l_plus_1_bar
                    cglp.setObjective(obj_expr, GRB.MAXIMIZE)

                    # 求解更新后的CGLP
                    # cglp.reset()
                    cglp.optimize()

                    # 如果找到一个有效的割平面
                    if cglp.status == GRB.OPTIMAL and cglp.ObjVal > TOL:
                        # 获取割平面系数 d_l, d_{l+1}, δ
                        delta_val = delta.X
                        d_l_val = cglp.getAttr('X', d_l)
                        d_l_plus_1_val = d_l_plus_1.X

                        # 构建割平面表达式
                        cut_expr = gp.quicksum(d_l_val[j] * x_vars[layer_idx][j] for j in range(prev_layer_size)) + \
                                   d_l_plus_1_val * x_vars[current_layer_num][neuron_idx]
                        
                        # 将割平面作为惰性约束添加到主模型
                        model.cbLazy(cut_expr >= delta_val)
                        # model.cbCut(cut_expr >= delta_val)
                        callback_stats['nCuts'] += 1
        
        callback_stats['time'] += time.time() - start_time
        

def build_model(model_name, target_class, epsilon):
    """
    Builds and solves the MILP model for the Optimal Sparse Input Features problem
    using a Big-M formulation for ReLUs.

    Parameters:
        model_name (str): name of the file containing model parameters.
        target_class (int): target class to maximize, can be 0-9.
        epsilon (float): maximum l1-norm defining perturbations.
    """
    model_path = Path('data/raw/osif/test') / model_name
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    nn_params = np.load(model_path)

    # 自动计算网络层数
    n_layers = 0
    while f'm{n_layers + 1}' in nn_params:
        n_layers += 1

    # --- BEGIN OPTIMIZATION MODEL ---

    model = gp.Model("OSIF_BigM")
    x = {}  # Post-activation variables
    y = {}
    z = {}  # Binary variables for ReLU status
    count = 0
    # --- Layer 0: Input Nodes ---

    input_size = nn_params['m1'].shape[1]
    x[0] = model.addVars(input_size, lb=0, ub=1, name="x_0")
    model.update()
    model.addConstr(gp.quicksum(x[0][i] for i in range(input_size)) <= epsilon, name="l1_norm_constraint")

    # --- Hidden and Output Layers ---

    for layer_idx in range(n_layers):
        current_layer_num = layer_idx + 1
        x[current_layer_num] = {}
        if layer_idx < n_layers - 1: # z variables only for ReLU layers
            z[current_layer_num] = {}
            y[current_layer_num] = {}

        m_layer = nn_params[f'm{current_layer_num}']
        b_layer = nn_params[f'b{current_layer_num}']
        
        for neuron_idx in range(m_layer.shape[0]):
            # Pre-activation expression: w*x + b
            pre_act_expr = gp.quicksum(x[layer_idx][j] * m_layer[neuron_idx, j] for j in range(m_layer.shape[1])) + b_layer[neuron_idx]
            m = m_layer[neuron_idx]
            b = b_layer[neuron_idx]
            # --- Calculate Bounds (L, U) for the pre-activation expression ---
            # This is crucial for setting the Big-M values
            L = sum(x[layer_idx][j].LB * max(0, m[j]) + x[layer_idx][j].UB * min(0, m[j]) for j in range(len(m))) + b
            U = sum(x[layer_idx][j].UB * max(0, m[j]) + x[layer_idx][j].LB * min(0, m[j]) for j in range(len(m))) + b

            # --- Final Layer (Linear) ---

            if layer_idx == n_layers - 1:
                x[current_layer_num][neuron_idx] = model.addVar(lb=L, ub=U, name=f'x_{current_layer_num}_{neuron_idx}')
                model.addConstr(x[current_layer_num][neuron_idx] == pre_act_expr, name=f'linear_{current_layer_num}_{neuron_idx}')
            
            # --- Hidden Layers (ReLU with Big-M) ---

            else:
                # Post-activation variable (output of ReLU)
                x[current_layer_num][neuron_idx] = model.addVar(lb=max(0, L), ub=max(0, U), name=f'x_{count}_l{layer_idx}n{neuron_idx}')
                
                # Binary variable `z`: 1 if neuron is ACTIVE, 0 if INACTIVE.
                z[current_layer_num][neuron_idx] = model.addVar(vtype=gp.GRB.BINARY, name=f'ind_disjunction_{count}_disjunct_2_l{layer_idx}n{neuron_idx}')
                y[current_layer_num][neuron_idx] = model.addVar(vtype=gp.GRB.BINARY, name=f'ind_disjunction_{count}_disjunct_1_l{layer_idx}n{neuron_idx}')
                # Big-M Constraints
                # If z=1 (active): pre_act >= 0, x_out = pre_act
                # Linearize: pre_act >= L * (1-z); x_out >= pre_act; x_out <= pre_act - L * (1-z)
                # If z=0 (inactive): pre_act <= 0, x_out = 0
                # Linearize: pre_act <= U * z; x_out <= U * z
                
                # # 2. pre_act >= L * y
                model.addConstr(pre_act_expr >= L * y[current_layer_num][neuron_idx], name=f'c_disjunction_{count}_disjunct_1.1_l{layer_idx}n{neuron_idx}')

                # # 4. x_out >= pre_act
                # model.addConstr(x[current_layer_num][neuron_idx] >= pre_act_expr, name=f'bigm_x_out_L_{current_layer_num}_{neuron_idx}')
                
                # pre_act - x_out <= U * y
                model.addConstr(pre_act_expr - x[current_layer_num][neuron_idx] <= 0.001 * y[current_layer_num][neuron_idx], name=f'c_disjunction_{count}_disjunct_1.2_l{layer_idx}n{neuron_idx}')
                
                # x_out <= pre_act - L * y
                model.addConstr(x[current_layer_num][neuron_idx] <= pre_act_expr - L * y[current_layer_num][neuron_idx], name=f'c_disjunction_{count}_disjunct_1.3_l{layer_idx}n{neuron_idx}')

                # pre_act <= U * z
                model.addConstr(pre_act_expr <= U * z[current_layer_num][neuron_idx], name=f'c_disjunction_{count}_disjunct_2.1_l{layer_idx}n{neuron_idx}')
                
                # 5. x_out <= U * z
                model.addConstr(x[current_layer_num][neuron_idx] <= U * z[current_layer_num][neuron_idx], name=f'c_disjunction_{count}_disjunct_2.2_l{layer_idx}n{neuron_idx}')
                
                model.addConstr(y[current_layer_num][neuron_idx] + z[current_layer_num][neuron_idx] == 1)

                count += 1

                model.update()

    # # --- 为分离割平面回调预构建CGLP模型 (每个神经元一个) ---
    model._cglp_data = {}
    for layer_idx in range(n_layers - 1):
        current_layer_num = layer_idx + 1
        prev_layer_size = nn_params[f'm{current_layer_num}'].shape[1]
        current_layer_size = nn_params[f'm{current_layer_num}'].shape[0]
        
        model._cglp_data[layer_idx] = {} # 为当前层的神经元创建一个字典

        for neuron_idx in range(current_layer_size):
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

            # 存储模型和关键变量的引用
            model._cglp_data[layer_idx][neuron_idx] = {
                'model': cglp,
                'vars': {'delta': delta, 'd_l': d_l, 'd_l_plus_1': d_l_plus_1}
            }

    model._nlayers = n_layers
    model._vars_x = x
    model._nn_params = nn_params
    model._once = False

    final_layer_idx = n_layers
    model.setObjective(x[final_layer_idx][target_class], gp.GRB.MAXIMIZE)
    return model

def main():
    """Main execution function."""
    # --- Parameters ---

    path = 'data/raw/osif/test'
    log_file = 'results/OSIF_test_evaluation2.csv'
    
    header = ["problem", "instance", "epsilon", "target", "algoName", "feasible", 
              "timeLimit", "timeCost", "ub", "lb", "status", "nCols", 
              "nRows", "nNodes", "timeCallback", "nCuts", "rootUb", 
              "rootLb", "rootTime", "rootGap", "rootTime%", "rootUb%", 
              "rootLb%", "nThreads", "gap%", "optimal"]
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # files.sort(key=lambda x: int(x.split("Cap")[0]))
    files_with_path = [os.path.join(path, f) for f in files]
    if not os.path.exists(log_file):
        init_log_file(log_file, header)

    strategies = {
        "Big-M": None,  # 30.51
        "Cut": seperating_disjunctive_callback # 28.315
        # "Cut_Root_100": seperating_disjunctive_callback_root
    }

    for i in range(len(files_with_path)): 
        for name, callback_func in strategies.items(): 
            result_dict = {key: None for key in header}
            target_class = 0
            epsilon = 10.0
            model_name = files[i]
            # file_path = files_with_path[i]
            print(f"Solving instance {model_name} with {name}, target={target_class} and epsilon={epsilon}")
            n_layers = 3
            
            model = build_model(model_name, target_class, epsilon)

            # model.setParam('MIPFocus', 3)
            # model.setParam('Cuts', 1)
            # model.setParam('PumpPasses', 100)
            # model.setParam('Method', 1) # Dual Simplex
            model.setParam('LazyConstraints', 1) # 启用惰性约束以使用回调
            # model.setParam('PreCrush', 1)
            model.setParam('TimeLimit', 1800)
            # model.setParam('LiftProjectCuts', 0)

            # 为回调准备统计数据容器
            callback_stats = {
                'startTime': None,
                'time': 0, 
                'nCuts': 0, 
                'rootUb': None, 
                'rootLb': None, 
                'rootTime': None
            }
            model._callback_stats = callback_stats
            callback_on = True

            # --- Solve Model ---
            t_start = time.time()
            model.optimize(callback_func) # 传递回调函数
            # model.optimize()
            solve_time = time.time() - t_start
            
            # --- 填充结果字典 ---
            result_dict['problem'] = "OSIF with ReLU-NNs"
            result_dict['instance'] = model_name
            result_dict['epsilon'] = epsilon
            result_dict['target'] = target_class
            result_dict['algoName'] = name
            result_dict['timeLimit'] = model.Params.TimeLimit
            result_dict['timeCost'] = solve_time
            result_dict['status'] = model.status
            result_dict['nCols'] = model.NumVars
            result_dict['nRows'] = model.NumConstrs
            result_dict['nNodes'] = model.NodeCount
            result_dict['nThreads'] = 16
            
            # 从回调中获取统计信息
            result_dict['timeCallback'] = callback_stats['time']
            result_dict['nCuts'] = callback_stats['nCuts']
            result_dict['rootUb'] = callback_stats['rootUb']
            result_dict['rootLb'] = callback_stats['rootLb']
            result_dict['rootTime'] = callback_stats['rootTime']

            # 根据 Gurobi 状态处理结果
            if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED, GRB.USER_OBJ_LIMIT]:
                result_dict['optimal'] = 1 if model.status == GRB.OPTIMAL else 0
                if model.SolCount > 0:
                    result_dict['feasible'] = 1
                    result_dict['ub'] = model.ObjBound
                    result_dict['lb'] = model.ObjVal
                    result_dict['gap%'] = model.MIPGap * 100
                else: # 没有找到可行解
                    result_dict['feasible'] = 0
                    result_dict['ub'] = model.ObjBound
                    result_dict['lb'] = float('-inf') if model.ModelSense == GRB.MAXIMIZE else float('inf')
                    result_dict['gap%'] = 100.0
            elif model.status == GRB.INFEASIBLE:
                result_dict['feasible'] = 0
                result_dict['optimal'] = 0
                result_dict['ub'] = float('inf')
                result_dict['lb'] = float('-inf')
                result_dict['gap%'] = None
            else:
                result_dict['feasible'] = 0
                result_dict['optimal'] = 0
                result_dict['ub'] = model.ObjBound if hasattr(model, 'ObjBound') else None
                result_dict['lb'] = None
                result_dict['gap%'] = None

            # 计算百分比和根节点gap
            if result_dict['timeCost'] > 0 and result_dict['rootTime'] is not None:
                result_dict['rootTime%'] = (result_dict['rootTime'] / result_dict['timeCost']) * 100
            
            if result_dict['ub'] is not None and result_dict['rootUb'] is not None and abs(result_dict['ub']) > 1e-9:
                result_dict['rootUb%'] = (result_dict['rootUb'] / result_dict['ub']) * 100
            
            if result_dict['lb'] is not None and result_dict['rootLb'] is not None and result_dict['lb'] not in [float('inf'), float('-inf')] and abs(result_dict['lb']) > 1e-9:
                result_dict['rootLb%'] = (result_dict['rootLb'] / result_dict['lb']) * 100
            
            if result_dict['rootUb'] is not None and result_dict['rootLb'] is not None and abs(result_dict['rootLb']) > 1e-9:
                result_dict['rootGap'] = abs(result_dict['rootUb'] - result_dict['rootLb']) / abs(result_dict['rootLb']) * 100

            log_performance(log_file, result_dict)

if __name__ == "__main__":
    main()
