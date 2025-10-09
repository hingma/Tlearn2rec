# 提供从CFLP的txt格式算例读取数据和建立模型的函数
import pyomo.environ as pyo
from pyomo.gdp import *
import log
import os
import time
import gurobipy as gp
from gurobipy import GRB
# import Bigm2Disjunct

class DataLoader:
    def __init__(self, file_path):
        self.n_facilities = 0
        self.n_clients = 0
        self.fixed_cost = 0
        self.capacity = 0

        self.demands = {}      # 客户ID: 需求
        self.transport_costs = {}  # (设施ID, 客户ID): 运输成本
        with open(file_path, 'r') as file:
            self.lines = file.readlines()

    def read_metadata(self):
        """
        Return:
            n_facilities
            n_clients
            fixed_cost
            capacity
        """
        line = self.lines[3].strip()
        meta_data = line.split()
        self.n_facilities = self.n_clients = int(meta_data[0])
        self.fixed_cost = int(meta_data[1])
        self.capacity = int(meta_data[2])
        return self.n_facilities, self.n_clients, self.fixed_cost, self.capacity

    def read_data(self):
        """
        Return:
            self.demands
            self.transport_costs
        """
        lines = self.lines[5:]

        for line in lines:
            line = line.strip()
            parts = line.split()
            if len(parts) >= 4:
                facility = int(parts[0])
                client = int(parts[1])
                cost = float(parts[2])
                demand = int(parts[3])
                self.transport_costs[facility, client] = cost
                self.demands[facility, client] = demand
        return self.demands, self.transport_costs

def build_model(data_loader: DataLoader) -> pyo.ConcreteModel:
    # load data
    n_facilities, n_customers, fixed_cost, capacity = data_loader.read_metadata()
    demands, costs = data_loader.read_data()
    customers = range(1, n_customers + 1)
    facilities = range(1, n_facilities + 1)

    model = pyo.ConcreteModel()
    # 集合
    model.J = pyo.Set(initialize=customers)   # 客户
    model.I = pyo.Set(initialize=facilities)  # 设施
    # 变量
    model.valid_pairs = pyo.Set(initialize=demands.keys(), dimen=2)
    model.x = pyo.Var(model.valid_pairs, within=pyo.Binary)  # 仅生成有效变量
    
    # 约束：每个客户的需求必须被满足
    def demand_constraint(m, j):
        return sum(model.x[i, j] for i in model.I if (i, j) in demands.keys()) == 1
    model.demand = pyo.Constraint(model.J, rule=demand_constraint)

    # 析取约束：设施开放或关闭的条件
    # 开放设施时的约束：分配量不超过容量
    def open_constraint(disjunct, i):
        model = disjunct.model()
        disjunct.c = pyo.Constraint(expr=sum(demands[i,j] * model.x[i,j] for j in model.J if (i, j) in demands.keys()) <= capacity)
    model.open_i = Disjunct(model.I, rule=open_constraint)

    # 关闭设施时的约束：分配量为0
    def close_constraint(disjunct, i):
        model = disjunct.model()
        disjunct.c = pyo.Constraint(expr=sum(model.x[i,j] for j in model.J if (i, j) in demands.keys()) <= 0)
    model.close_i = Disjunct(model.I, rule=close_constraint)

    def disjunction_rule(model, i):

        # close_cond = [model.y[j] == 0]
        # for i in model.I:
        #     if (i, j) in demands.keys():
        #         close_cond.append(model.x[i, j] == 0)
        return [model.open_i[i], model.close_i[i]]

    model.facility_disjunction = Disjunction(model.I, rule=disjunction_rule)

    # 目标函数：最小化总成本
    def objective_rule(model):
        return sum(fixed_cost * model.open_i[i].indicator_var.get_associated_binary() for i in model.I) + \
            sum(costs[i,j] * model.x[i,j] for i in model.I for j in model.J if (i, j) in demands.keys())
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model

def build_gurobi_model(data_loader):
    # 加载数据
    n_facilities, n_customers, fixed_cost, capacity = data_loader.read_metadata()
    demands, costs = data_loader.read_data()
    customers = range(1, n_customers + 1)
    facilities = range(1, n_facilities + 1)

    # 创建Gurobi模型
    model = gp.Model("CFLP")

    # 定义变量
    y = model.addVars(facilities, vtype=gp.GRB.BINARY, name="y")  # 设施开放状态
    z = model.addVars(facilities, vtype=gp.GRB.BINARY, name="z")  # 设施关闭状态
    
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
        y[i].VarName = f"ind_disjunction_{i}_disjunct_1"
        z[i].VarName = f"ind_disjunction_{i}_disjunct_2"
        model.addConstr(
            sum(demands[i,j] * x[i,j] for j in customers if (i,j) in valid_pairs) <= capacity * y[i],
            name=f"c_disjunction_{i}_diajunct_1"
        )
        model.addConstr(
            sum(x[i,j] for j in customers if (i,j) in valid_pairs) <= min(sum(1 for j in customers if (i,j) in valid_pairs), capacity) * (1 - z[i]),
            name=f"c_disjunction_{i}_disjunct_2"
        )
        model.addConstr(y[i] + z[i] == 1)
        model.update()
       

    return model
# 主程序
def main():
    path = '/home/bhz/Code/experiment/python/data/CFLP/'  # 数据集路径
    log_file = '/home/bhz/Code/experiment/python/results/CFLP_results.csv'
    # 只获取文件（不包括子文件夹）
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # 获取包含完整路径的文件列表
    files_with_path = [os.path.join(path, f) for f in files]
    
    # 三种松弛方式
    # relaxations = ['hull', 'bigm', 'bigm(M=Capacity)', 'mbigm']
    relaxations = ['between_steps']
    if not os.path.exists(log_file):
        log.init_log_file(log_file)

    for i in range(len(files_with_path)):
        for relaxation in relaxations:
            print(f"正在求解实例 {files[i]}  relaxation {relaxation}")
            instance = files[i]
            file_path = files_with_path[i]
            data_loader = DataLoader(file_path)
            model = build_model(data_loader) #, bigM={None: 300}
            # 记录转换时间
            t_start = time.time()
            if relaxation == 'bigm(M=Capacity)':
                pyo.TransformationFactory(f'gdp.bigm').apply_to(model, bigM={None: data_loader.capacity})
            else:
                pyo.TransformationFactory(f'gdp.{relaxation}').apply_to(model, num_partitions=100)
            transform_time = time.time() - t_start
            # model.write(f"/home/bhz/Code/experiment/model_bigm2.mps", format="mps")

            solver = pyo.SolverFactory('gurobi')
            solver.options['TimeLimit'] = 180
            solver.options['Seed'] = 42
            solver.options['Threads'] = 16
            # solver.options['ConcurrentMIP'] = 1

            t_start = time.time()
            results = solver.solve(model, tee=True)
            solve_time = time.time() - t_start
            # 记录结果
            status = results.solver.status
            termination = results.solver.termination_condition
            objective = results['Problem'][0]['Upper bound']
            log.log_performance(
                log_file,
                "CFLP",
                instance,
                f"between_steps-pyomo(P=100)",
                transform_time=0,
                solve_time=solve_time,
                gap=0,
                objective=objective,
                status=status,
                termination=termination,
                error_msg=''
            )

            # 保存lp文件
            # model.write(f"model_files/CFLP_{instance}_pyomo.lp", format="lp")

        # transformer = Bigm2Disjunct.BigmTransformer()
        # model2 = transformer.build_pyomo_model('/home/bhz/experiment/model_bigm2.mps')
        # model2 = build_model(data_loader)
        # pyo.TransformationFactory('gdp.hull').apply_to(model2)
        # solver = pyo.SolverFactory('gurobi')
        # solver.options['TimeLimit'] = 180
        # t_start2 = time.time()
        # results2 = solver.solve(model2, tee=True)
        # solve_time2 = time.time() - t_start2
        # # 记录结果
        # status = results2.solver.status
        # termination = results2.solver.termination_condition
        # objective = results2['Problem'][0]['Upper bound']
        # log.log_performance(
        #     log_file,
        #     "CFLP",
        #     instance,
        #     "hull",
        #     0,
        #     solve_time2,
        #     gap=0,
        #     objective=objective,
        #     status=status,
        #     termination=termination,
        #     error_msg=''
        # )

        # # 进行求解并记录
        # log.solve_with_monitoring(log_file, model, "CFLP", instance, time_limit=300)
    
    # data_loader = DataLoader("/home/bhz/experiment/python/CFLP_data/toy_instance.txt")
    # model = build_model(data_loader)
    # results = solve_model(model)
    
if __name__ == "__main__":
    # main()
    file_path = '/home/bhz/Code/experiment/python/data/CFLP/2Cap10.txt'
    data_loader = DataLoader(file_path)
    model = build_model(data_loader)
    pyo.TransformationFactory(f'gdp.bigm').apply_to(model)
    # pyo.TransformationFactory('gdp.hull').apply_to(model)
    solver = pyo.SolverFactory('gurobi')
    results = solver.solve(model, tee=True)
    model.write("model_files/toymodel_bigm.lp", format="lp") 