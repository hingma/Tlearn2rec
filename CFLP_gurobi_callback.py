# 完整的使用CGLP加速CFLP求解的代码，与其他代码无直接关联
import log
import os
import time
import gurobipy as gp
from gurobipy import GRB
from CFLP import DataLoader
from utilities import CFLP_control_callback as control_callback
from utilities import CFLP_seperating_disjunctive_callback_once as seperating_disjunctive_callback_once
from utilities import CFLP_seperating_disjunctive_callback_root as seperating_disjunctive_callback_root



def lift_and_project_callback(model, where):
    """
    Gurobi 回调函数，用于在MIP节点上添加 lift-and-project cuts。
    """
    # 仅在MIP节点找到LP最优解时运行
    if where == gp.GRB.Callback.MIPNODE and model.cbGet(gp.GRB.Callback.MIPNODE_STATUS) == gp.GRB.OPTIMAL:
        # count = model.cbGet(gp.GRB.Callback.MIPNODE_NODCNT)
        # if count > 10:
        #     return
        # 获取LP松弛解的变量值
        y_vals = model.cbGetNodeRel(model._vars_y)
        x_vals = model.cbGetNodeRel(model._vars_x)

        n_cuts_added = 0
        facilities = model._facilities
        facility_to_customers = model._facility_to_customers
        
        # 遍历所有分数解的设施
        for i in facilities:
            # 只关心那些y[i]为分数的设施
            if 0.001 < y_vals[i] < 0.999:
                # 高效地遍历连接到该设施的客户
                for j in facility_to_customers.get(i, []):
                    # 如果 x[i,j] > y[i]，则违反了约束
                    if x_vals[i, j] > y_vals[i] + 1e-6: # 使用一个小的容差
                        # 添加懒惰约束 (lazy cut)
                        model.cbLazy(model._vars_x[i, j] <= model._vars_y[i])
                        n_cuts_added += 1
        
        if n_cuts_added > 0:
            print(f"Callback: 在节点上添加了 {n_cuts_added} 个 L&P cuts。")

def build_model(data_loader):
    # 加载数据
    n_facilities, n_customers, fixed_cost, capacity = data_loader.read_metadata()
    demands, costs = data_loader.read_data()
    customers = range(1, n_customers + 1)
    facilities = range(1, n_facilities + 1)

    # 创建Gurobi模型
    model = gp.Model("CFLP")

    # 定义变量
    y = model.addVars(facilities, vtype=gp.GRB.BINARY, name="y")  # 设施开放状态
    
    # 仅为有效配对创建x变量
    valid_pairs = list(demands.keys())
    x = model.addVars(valid_pairs, vtype=gp.GRB.BINARY, name="x")  # 客户分配

    # 为回调函数创建从设施到其有效客户的映射，以提高效率
    facility_to_customers = {i: [] for i in facilities}
    for i, j in valid_pairs:
        facility_to_customers[i].append(j)

    model._vars_y = y
    model._vars_x = x
    model._facilities = facilities
    model._facility_to_customers = facility_to_customers
    model._demands = demands
    model._capacity = capacity
    model._once = False
    model._call_num = 0
    model.update()

    # 目标函数：最小化总成本
    model.setObjective(
        sum(fixed_cost * y[i] for i in facilities) + 
        sum(costs[i,j] * x[i,j] for i,j in valid_pairs),
        gp.GRB.MINIMIZE
    )

    # 约束：每个客户的需求必须被满足
    for j in customers:
        model.addConstr(
            sum(x[i,j] for i in facilities if (i,j) in valid_pairs) == 1,
            name=f"demand_{j}"
        )

    # Hull Reformulation
    # for i in facilities:
    #     model.addConstr(
    #         sum(demands[i,j] * x[i,j] for j in customers if (i,j) in valid_pairs) <= capacity * y[i],
    #     )
    #     for j in customers:
    #         if (i,j) in valid_pairs:
    #             model.addConstr(x[i,j] <= y[i])
    
    # big-M Reformulation
    for i in facilities:
        model.addConstr(
            sum(demands[i,j] * x[i,j] for j in customers if (i,j) in valid_pairs) <= capacity * y[i],
            name=f"capacity_indicator_{i}"
        )
        # model.addConstr(
        #     sum(x[i,j] for j in customers if (i,j) in valid_pairs) <= min(sum(1 for j in customers if (i,j) in valid_pairs), capacity) * y[i],
        #     name=f"assignment_indicator_{j}"
        # )
    

        # --- 为回调预先构建CGLP模型 ---
    # 这可以显著提高回调性能，因为它避免了在每个节点上重复创建模型。
    model._cglp_models = {}
    for i in facilities:
        customers_for_i = facility_to_customers.get(i, [])
        if not customers_for_i:
            continue

        cglp = gp.Model(f"CGLP_{i}")
        cglp.setParam('OutputFlag', 0)
        # cglp.setParam('Method', 0)  # Primal Simplex for CGLP

        # CGLP 变量 (割平面系数)
        alpha = cglp.addVars(customers_for_i, lb=-1, ub=1, name="alpha")
        beta = cglp.addVar(lb=-1, ub=1, name="beta")
        delta = cglp.addVar(lb=-GRB.INFINITY, name="delta")

        # CGLP中的对偶变量
        u = cglp.addVar(lb=0, name="u")
        u_2 = cglp.addVar(lb=-GRB.INFINITY, name="u_2")
        v = cglp.addVar(lb=0, name="v")
        v_2 = cglp.addVar(lb=-GRB.INFINITY, name="v_2")

        # CGLP 约束 (定义有效割平面的空间)
        for j in customers_for_i:
            cglp.addConstr(alpha[j] >= -u * demands[i, j])
            cglp.addConstr(alpha[j] >= -v)
        cglp.addConstr(delta <= -capacity * u + u_2)
        cglp.addConstr(beta >= u_2)
        cglp.addConstr(delta <= 0)
        cglp.addConstr(beta >= v_2)

        # 存储模型和变量句柄以供回调使用
        model._cglp_models[i] = {'model': cglp, 'alpha': alpha, 'beta': beta, 'delta': delta}

    

    return model

def solve_and_log(model, log_file, instance_name, strategy_name, callback=None):
    """
    使用指定的策略求解模型，并记录性能指标。

    Args:
        model (gp.Model): 要解决的Gurobi模型。
        log_file (str): CSV日志文件的路径。
        instance_name (str): 当前求解的实例名称。
        strategy_name (str): 正在使用的求解策略的名称（例如 "Gurobi Default"）。
        callback (function, optional): 要在优化期间使用的回调函数。
    """
    print(f"  - 使用策略 '{strategy_name}' 进行求解...")
    
    # 重置模型并初始化自定义指标
    model.reset()
    model._cuts_added = 0
    model._initial_root_bound = None
    model._once = False
    model.update()

    # 设置模型参数
    model.setParam('TimeLimit', 300)
    model.setParam('Seed', 123)
    if callback:
        model.setParam('LazyConstraints', 1)
        # PreCrush=1 告知 Gurobi 我们将添加用户割平面，这有助于保留原始模型结构
        # model.setParam('PreCrush', 1)
        # model.setParam('Cuts', 0)
    else:
        # 如果不使用回调，可以恢复Gurobi的默认设置
        model.setParam('LazyConstraints', 0)
        # model.setParam('PreCrush', 1)

    t_start = time.time()
    model.optimize(callback)
    solve_time = time.time() - t_start

    # 收集指标
    final_objective = model.ObjVal if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else 'N/A'
    final_gap = model.MIPGap if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else 'N/A'
    total_nodes = model.NodeCount if hasattr(model, 'NodeCount') else 'N/A'
    status = model.status

    # 收集根节点评估指标
    cuts_added = model._cuts_added
    initial_root_bound = model._initial_root_bound
    final_root_bound = model.getAttr('ObjBoundC') if hasattr(model, 'ObjBoundC') else None
    
    root_bound_improvement = 'N/A'
    if initial_root_bound is not None and final_root_bound is not None:
        # 对于最小化问题，界是越大越好
        root_bound_improvement = final_root_bound - initial_root_bound

    # 记录到CSV
    log.log_performance(
        log_file, "CFLP", instance_name, strategy_name,
        transform_time=0, solve_time=solve_time, gap=final_gap, objective=final_objective,
        status=status, termination=model.status, nodes=total_nodes,
        cuts_added=cuts_added, initial_root_bound=initial_root_bound, final_root_bound=final_root_bound,
        root_bound_improvement=root_bound_improvement
    )
    print(f"    完成. 时间: {solve_time:.2f}s, 目标值: {final_objective}, Gap: {final_gap}, 节点数: {total_nodes}")
    if callback:
        print(f"    根节点评估: 添加了 {cuts_added} 个割平面, 界从 {initial_root_bound} 提升到 {final_root_bound} (提升: {root_bound_improvement})")


def print_solution(model):
    print("设施开启情况：")
    for j in model.getVars():
        if j.varName.startswith("y") and j.x > 0.5:
            print(f"设施 {j.varName[2:-1]} 已开启")
    
    print("\n客户分配：")
    for j in model.getVars():
        if j.varName.startswith("x") and j.x > 0.5:
            print(f"客户 {j.varName[2:].split(',')[0]} 分配给设施 {j.varName[2:].split(',')[1][:-1]}")


# 主程序
if __name__ == "__main__":
    path = 'data/raw/facilities/transfer_100_100'  # 数据集路径
    log_file = 'results/CFLP_100_100_evaluation.csv'
    
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files_with_path = [os.path.join(path, f) for f in files]
    
    if not os.path.exists(log_file):
        log.init_log_file(log_file)

    # 定义要测试的策略
    strategies = {
        "Gurobi_Default": control_callback,  # 30.51
        "Cut_Root_Once": seperating_disjunctive_callback_once # 28.315
        # "Cut_Root_100": seperating_disjunctive_callback_root
    }

    for file_path in files_with_path:
        instance_name = os.path.basename(file_path)
        print(f"\n正在求解实例 {instance_name}")
        
        data_loader = DataLoader(str(file_path))
        model = build_model(data_loader)

        for name, callback_func in strategies.items():
            solve_and_log(model, log_file, instance_name, name, callback_func)

    print(f"\n评估完成。结果已保存到 {log_file}")
