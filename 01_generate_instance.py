# 生成算例原始格式的数据，储存在data/raw中
# 使用方法：python 01_generate_instance.py [problem]（现在只有facilities）
import os
import argparse
import numpy as np
import utilities
import shutil
import config

def generate_capacited_facility_location(rng, filename, dimension, ratio):
    """
    Generates a Capacitated Facility Location problem instance.

    This generator creates a sparse version of the problem where each client can only
    be served by a subset of nearby facilities. It also introduces variability in
    demands for each client-facility pair. Crucially, it guarantees that at least
    one feasible solution exists.

    - Service Sparsity: Each client can be served by its `K_nearest` facilities.
    - Demand Variability: Demand is not just client-specific, but client-facility pair specific.
    - Guaranteed Feasibility: The capacity of facilities is set to be large enough
      to accommodate a baseline feasible solution where every client is served by
      its closest facility.

    Saves it as a custom TXT file.

    Parameters
    ----------
    rng : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    dimension : int
        The number of customers and facilities.
    ratio : float
        A factor influencing facility capacity, related to the total demand.
    """
    dimension = rng.randint(max(1, dimension - 30), dimension + 31)
    # 1. Generate client and facility coordinates
    c_x = rng.rand(dimension)
    c_y = rng.rand(dimension)

    # 设施和客户使用相同的坐标，模拟它们位于同一区域
    f_x = c_x
    f_y = c_y

    # 2. 为每个客户生成一个固定的需求 (demand_j)
    demands = rng.randint(5, 35 + 1, size=dimension)

    # 3. Calculate all-pairs distances between clients and facilities
    # Trick：使用广播操作将一维坐标数组转换为二维列向量和行向量，Numpy执行效率高
    # 得到的矩阵client为列，facility为行
    distances = np.sqrt(
        (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2
        + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2
    )

    # 4. Construct a baseline feasible solution to guarantee one exists.
    #    Assign each client to its closest facility.
    # argmin返回最小值对于的索引，axis=1指定沿着行操作
    closest_facility_indices = np.argmin(distances, axis=1)
    facility_loads = np.zeros(dimension)
    for i in range(dimension):
        facility_loads[closest_facility_indices[i]] += demands[i]
    
    max_load = facility_loads.max()

    # 5. Set a single capacity for all facilities.
    #    The capacity must be large enough for the baseline feasible solution.
    #    We also incorporate the original logic which uses the ratio parameter.
    total_demand = demands.sum()
    capacity_from_ratio = total_demand * ratio / dimension
    capacity = int(max(max_load, capacity_from_ratio))
    # Ensure capacity is at least 1
    if capacity == 0:
        capacity = 1

    # 6. 为每个设施生成不同的固定成本
    fixed_costs = rng.randint(100, 110 + 1, size=dimension) * np.sqrt(capacity) + rng.randint(0, 90 + 1, size=dimension)
    fixed_costs = fixed_costs.astype(int)

    # 7. Determine serviceable pairs and generate specific data for each.
    #    为了让问题更稀疏、更容易，每个客户只能由其最近的20%的设施服务
    K_nearest = int(0.1 * dimension)
    problem_lines = []

    for i in range(dimension):  # For each client
        facility_distances = distances[i, :]
        # Get indices of the K nearest facilities
        nearest_facility_indices = np.argsort(facility_distances)[:K_nearest] # 升序排列，取前K个

        for j in nearest_facility_indices:  # For each of the K nearest facilities
            demand_j = demands[i]

            # 运输成本基于距离和客户需求
            cost_ij = distances[i, j] * 10 * demand_j

            # Facility is j+1, client is i+1 (1-based indexing for the file)
            problem_lines.append(f"{j + 1}\t{i + 1}\t{cost_ij:.1f}\t{demand_j}\n")

    # 8. Write the complete problem to the file.
    with open(filename, 'w') as file:
        file.write("\nCode\n")
        file.write("Dimension   Fixed Cost    Capacity\n")
        # 注意：虽然我们为每个设施生成了不同成本，但文件格式似乎只支持一个全局固定成本。
        # 我们将使用平均成本写入文件。解析代码（CFLP.py）需要相应调整才能支持多成本。
        # 目前，为了保持兼容性，我们写入平均值。
        file.write(f"{dimension}\t{int(np.mean(fixed_costs))}\t{capacity}\n")
        file.write("Facility\tClient\tTransportation Cost\tDemand\n")
        for line in problem_lines:
            file.write(line)

def generate_OSIF(rng, filename, input_dim, hidden_dims, output_dim):
    """
    Generates an Optimal Sparse Input Feature (OSIF) problem instance.

    This function programmatically creates a feasible OSIF instance by:
    1. Defining a ReLU neural network structure.
    2. Randomly initializing its weights and biases.
    
    The resulting network parameters and target output are saved to a .npz file.

    Parameters
    ----------
    rng : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the .npz file to save.
    input_dim : int
        The dimension of the input layer.
    hidden_dims : list of int
        A list containing the dimensions of each hidden layer.
    output_dim : int
        The dimension of the output layer.
    """
    nn_params = {}
    layer_dims = [input_dim] + hidden_dims + [output_dim]

    # 1. Randomly generate network weights and biases
    for i in range(len(layer_dims) - 1):
        in_features = layer_dims[i]
        out_features = layer_dims[i+1]

        # Use He initialization for weights, good for ReLU networks
        std_dev = np.sqrt(2. / in_features)
        nn_params[f'm{i+1}'] = rng.normal(0, std_dev, (out_features, in_features)).astype(np.float32)
        nn_params[f'b{i+1}'] = rng.rand(out_features).astype(np.float32)

    np.savez(filename, **nn_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['facilities', 'osif'],
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed (default 0).',
        type=utilities.valid_seed,
        default=0,
    )
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # 从配置文件加载实例生成参数
    generation_specs = config.INSTANCE_GEN.get(args.problem, {})
    base_dir = config.RAW_DATA_DIR / args.problem

    print(f"Generating instances for problem: {args.problem}")

    for name, specs in generation_specs.items():
        
        n_instances = specs['n_instances']
        if n_instances == 0:
            continue

        # 为文件夹名称添加维度信息以更好地区分
        if args.problem == 'facilities':
            dir_name = name if not name.startswith('transfer') else f"{name}_{specs['dimension']}"
        elif args.problem == 'osif':
            dir_name = name
        else:
            dir_name = name

        lp_dir = base_dir / dir_name

        print(f"{n_instances} instances in {lp_dir}")

        if not specs.get('overwrite', False) and lp_dir.exists():
            print(f"  skipping, directory already exists and overwrite is False.")
            continue

        if lp_dir.exists():
            shutil.rmtree(lp_dir)
        os.makedirs(lp_dir, exist_ok=True)

        for i in range(n_instances):
            print(f"  generating instance {i+1}/{n_instances}...")
            if args.problem == 'facilities':
                filename = lp_dir / f'instance_{i+1}.txt'
                generate_capacited_facility_location(rng, str(filename), dimension=specs['dimension'], ratio=specs['ratio'])
            elif args.problem == 'osif':
                filename = lp_dir / f'instance_{i+1}.npz'
                generate_OSIF(rng, str(filename), input_dim=specs['input_dim'], hidden_dims=specs['hidden_dims'], output_dim=specs['output_dim'])

    print("done.")
    
    if args.problem == 'facilities':
        # 从配置文件加载实例生成参数
        generation_specs = config.INSTANCE_GEN.get(args.problem, {})
        base_dir = config.RAW_DATA_DIR / args.problem

        print(f"Generating instances for problem: {args.problem}")

        for name, specs in generation_specs.items():
            
            n_instances = specs['n_instances']
            if n_instances == 0:
                continue

            dir_name = name if not name.startswith('transfer') else f"{name}_{specs['dimension']}"
            lp_dir = base_dir / dir_name

            print(f"{n_instances} instances in {lp_dir}")

            if not specs['overwrite'] and lp_dir.exists():
                continue

            if lp_dir.exists():
                shutil.rmtree(lp_dir)
            os.makedirs(lp_dir, exist_ok=True)

            for i in range(n_instances):
                filename = lp_dir / f'instance_{i+1}.txt'
                print(f"  generating file {filename} ...")
                generate_capacited_facility_location(rng, str(filename), dimension=specs['dimension'], ratio=specs['ratio'])

        print("done.")
