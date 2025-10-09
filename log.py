import time
import csv
import pyomo.environ as pyo
from pyomo.opt import SolverStatus as pyo_SolverStatus, TerminationCondition as pyo_TerminationCondition
import math
import gurobipy as gp

# Gurobi状态码到可读字符串的映射
GUROBI_STATUS_MAP = {
    gp.GRB.LOADED: "LOADED",
    gp.GRB.OPTIMAL: "OPTIMAL",
    gp.GRB.INFEASIBLE: "INFEASIBLE",
    gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
    gp.GRB.UNBOUNDED: "UNBOUNDED",
    gp.GRB.CUTOFF: "CUTOFF",
    gp.GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    gp.GRB.NODE_LIMIT: "NODE_LIMIT",
    gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    gp.GRB.INTERRUPTED: "INTERRUPTED",
    gp.GRB.NUMERIC: "NUMERIC",
    gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    gp.GRB.INPROGRESS: "INPROGRESS",
    gp.GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}
 
# 初始化日志文件（如果不存在）
def init_log_file(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "Problem", "Instance", "Relaxation", "TransformTime(s)", "SolveTime(s)",
            "Final_Gap(%)", "Final_Objective", "Total_Nodes",
            "Cuts_Added", "Initial_Root_Bound", "Final_Root_Bound", "Root_Bound_Improvement",
            "SolverStatus", "TerminationCondition"
        ]
        writer.writerow(header)

def _format_status(status_code):
    """将Gurobi状态码转换为可读字符串"""
    return GUROBI_STATUS_MAP.get(status_code, str(status_code))

# 记录单次运行结果
def log_performance(log_file, problem, instance_name, relaxation, transform_time, solve_time,
                   gap, objective, status, termination, nodes="N/A", cuts_added="N/A",
                   initial_root_bound="N/A", final_root_bound="N/A", root_bound_improvement="N/A"):
    try:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                problem,
                instance_name,
                relaxation,
                f"{transform_time:.2f}",
                f"{solve_time:.2f}",
                f"{gap*100:.4f}" if isinstance(gap, (int, float)) else "N/A",
                f"{objective:.4f}" if isinstance(objective, (int, float)) else "N/A",
                nodes,
                cuts_added,
                f"{initial_root_bound:.4f}" if isinstance(initial_root_bound, (int, float)) else "N/A",
                f"{final_root_bound:.4f}" if isinstance(final_root_bound, (int, float)) else "N/A",
                f"{root_bound_improvement:.4f}" if isinstance(root_bound_improvement, (int, float)) else "N/A",
                _format_status(status),
                _format_status(termination)
            ]
            writer.writerow(row)
    except Exception as e:
        print(f"写入文件失败: {e}")

# 带性能监控的求解流程
def solve_with_monitoring(log_file, model, problem, instance_name, relaxation, time_limit=180):
    """
    Solve a Pyomo model with performance monitoring, including transformation time,
    solve time, objective, gap, status, and termination condition.
    Captures Gurobi gap when time limit is reached.

    Args:
        log_file (str): Path to log file (not used in this snippet, but kept from original).
        model (pyo.Model): The Pyomo model to solve.
        problem (str): Name of the problem (for logging/tracking, not used in this snippet).
        instance_name (str): Name of the instance (for logging/tracking, not used in this snippet).
        relaxation (str): The name of the GDP relaxation method to apply.
        time_limit (int): Time limit in seconds for the solver.
    """  
    transform_time = None
    solve_time = None
    gap = None
    objective = None
    status = None
    termination = None
    error_msg = ""
    ub = None # Best feasible objective (Upper Bound for minimization)
    lb = None # Best bound (Lower Bound for minimization)

    try:
        # 转换阶段计时 (此部分主要用于Pyomo，在Gurobi脚本中transform_time为0)
        t_start = time.time()
        pyo.TransformationFactory('gdp.'+ relaxation).apply_to(model)
        transform_time = time.time() - t_start

        # 保存为LP文件 
        model.write(f"model_{relaxation}.lp", format="lp") 

        # 求解阶段计时
        solver = pyo.SolverFactory('gurobi')
        solver.options['TimeLimit'] = time_limit  # 设置求解时间限制
        
        t_start = time.time()
        results = solver.solve(model, tee=True)
        solve_time = time.time() - t_start

        # 获取求解状态
        status = results.solver.status # Pyomo SolverStatus
        termination = results.solver.termination_condition # Pyomo TerminationCondition

        # --- 获取 objective, gap, ub, lb ---

        # {'Problem': [{'Name': 'x1', 'Lower bound': 1022.9999999999911, 'Upper bound': 1320.0, 'Number of objectives': 1, 'Number of constraints': 14400, 'Number of variables': 9676, 'Number of binary variables': 3150, 'Number of integer variables': 3150, 'Number of continuous variables': 6526, 'Number of nonzeros': 35100, 'Sense': 'minimize'}], 'Solver': [{'Status': 'aborted', 'Return code': 0, 'Message': 'Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.', 'Termination condition': 'maxTimeLimit', 'Termination message': 'Optimization terminated because the time expended exceeded the value specified in the TimeLimit parameter.', 'Wall time': 30.006511926651, 'Error rc': 0}], 'Solution': [OrderedDict({'number of solutions': 1, 'number of solutions displayed': 1}), {'Status': 'unknown', 'Problem': {}, 'Objective': {}, 'Variable': {}, 'Constraint': {}}]}
        ub = results['Problem'][0]['Upper bound']
        objective = ub

        lb = results['Problem'][0]['Lower bound']

        # Calculate GAP based on termination condition and available bounds
        if termination == pyo_TerminationCondition.optimal:
            # For optimal solution, gap is 0 and objective is the optimal value
            gap = 0.0
            # Redundant, but ensures objective is set if not already from best_feasible
            if objective is None and pyo.value(model.obj) is not None:
                 objective = pyo.value(model.obj)
        
        elif termination == pyo_TerminationCondition.maxTimeLimit:
             # Time limit reached. Calculate gap if both feasible and bound are available.
             if ub is not None and lb is not None:
                 denominator = abs(ub)
                 if denominator > 1e-9: # Avoid division by zero or near-zero
                    if model.obj.sense == pyo.minimize:
                         # Gap = (Feasible - Bound) / |Feasible| = (UB - LB) / |UB|
                         gap = (ub - lb) / denominator * 100.0
                    elif model.obj.sense == pyo.maximize:
                         # Gap = (Bound - Feasible) / |Feasible| = (LB - UB) / |UB|
                         gap = (lb - ub) / denominator * 100.0
                    # Gap is usually reported as a non-negative percentage
                    gap = abs(gap)
                 else:
                    # Feasible objective is zero or near zero, gap is often considered undefined/infinite
                    gap = float('inf') if ub == 0 and lb != 0 else None # Handle edge case of 0 objective

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        # In case of error, status and termination might not be set, ensure they are represented
        if status is None: status = pyo_SolverStatus.error
        if termination is None: termination = pyo_TerminationCondition.error
    
    # 记录结果
    log_performance(
        log_file,
        problem,
        instance_name,
        relaxation,
        transform_time=transform_time or 0,
        solve_time=solve_time or 0,
        gap=gap,
        objective=objective,
        status=status,
        termination=termination,
    )
