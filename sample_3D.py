import math
import random


def poisson_disk_sampling_3d(width, height, depth, r, k=30, seed=None):
    random.seed(seed)

    # 1. 初始化网格
    # 在 3D 中，每个格子的大小是 r / sqrt(3)。这样确保格子里最多只有一个点。
    cell_size = r / math.sqrt(3)

    # 计算网格在 x, y, z 方向的数量
    cols = int(math.ceil(width / cell_size))
    rows = int(math.ceil(height / cell_size))
    layers = int(math.ceil(depth / cell_size))

    # 用一个一维数组来模拟三维网格，存储点的索引。初始值 -1 表示格子是空的。
    # 数组大小是 cols * rows * layers
    grid = [-1] * (cols * rows * layers)

    # 用来存放最终生成的点
    samples = []
    # 用来存放“活跃”点的索引（也就是周围可能还能放得下新点的那些点）
    active_list = []

    # 2. 随机生成第一个点
    x = random.random() * width
    y = random.random() * height
    z = random.random() * depth
    first_point = (x, y, z)

    # 把第一个点存起来
    samples.append(first_point)
    active_list.append(0)  # 索引是 0

    # 把第一个点记录在网格中
    col = int(x / cell_size)
    row = int(y / cell_size)
    layer = int(z / cell_size)

    # 计算一维索引: (layer * rows + row) * cols + col
    grid[(layer * rows + row) * cols + col] = 0

    # 3. 主循环：只要还有活跃点，就继续找
    while len(active_list) > 0:
        # 随机选一个活跃点
        rand_index = random.randint(0, len(active_list) - 1)
        point_index = active_list[rand_index]
        px, py, pz = samples[point_index]

        found_new_point = False

        # 尝试在这个点周围生成 k 个随机点
        for _ in range(k):
            # --- 生成随机球壳内的点 (半径 r ~ 2r) ---

            # 第一步：随机通用的方向
            # 使用高斯分布是生成球面上均匀随机点的标准方法
            dx = random.gauss(0, 1)
            dy = random.gauss(0, 1)
            dz = random.gauss(0, 1)

            # 归一化（变成长度为 1 的向量）
            d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d_len == 0:
                continue
            dx /= d_len
            dy /= d_len
            dz /= d_len

            # 第二步：随机半径 [r, 2r]
            # 为了让点在体积内均匀分布，半径的立方应该是均匀分布的
            # 也就是从 r^3 到 (2r)^3 = 8r^3 之间取值
            u = random.random()
            # radius = (r^3 + u * (8r^3 - r^3))^(1/3) = r * (1 + 7u)^(1/3)
            radius = r * (1 + 7 * u) ** (1 / 3)

            # 计算新点的坐标
            new_x = px + dx * radius
            new_y = py + dy * radius
            new_z = pz + dz * radius

            # 检查 1: 有没有超出画布边界？
            if not (0 <= new_x < width and 0 <= new_y < height and 0 <= new_z < depth):
                continue

            # 检查 2: 有没有离其他点太近？
            new_col = int(new_x / cell_size)
            new_row = int(new_y / cell_size)
            new_layer = int(new_z / cell_size)

            too_close = False
            r_sq = r * r

            # 确定循环范围 5x5x5
            # 注意不要超出网格边界
            layer_min = max(0, new_layer - 2)
            layer_max = min(layers, new_layer + 3)
            row_min = max(0, new_row - 2)
            row_max = min(rows, new_row + 3)
            col_min = max(0, new_col - 2)
            col_max = min(cols, new_col + 3)

            for l_idx in range(layer_min, layer_max):
                for r_idx in range(row_min, row_max):
                    for c_idx in range(col_min, col_max):
                        # 获取网格索引
                        neighbor_idx = grid[(l_idx * rows + r_idx) * cols + c_idx]
                        if neighbor_idx != -1:
                            # 如果格子里有点，计算距离
                            ex, ey, ez = samples[neighbor_idx]
                            dist_sq = (ex - new_x) ** 2 + (ey - new_y) ** 2 + (ez - new_z) ** 2
                            if dist_sq < r_sq:
                                too_close = True
                                break
                    if too_close:
                        break
                if too_close:
                    break

            # 如果不拥挤，记录这个新点
            if not too_close:
                found_new_point = True
                samples.append((new_x, new_y, new_z))
                new_idx = len(samples) - 1
                active_list.append(new_idx)
                grid[(new_layer * rows + new_row) * cols + new_col] = new_idx
                break  # 成功找到一个点，跳出 k 次循环

        # 如果试了 k 次都没成功，说明这个点周围满了
        if not found_new_point:
            # 把它从活跃列表里移除
            active_list.pop(rand_index)

    return samples


if __name__ == "__main__":
    # 配置参数
    W, H, D = 1.0, 1.0, 1.0
    R = 0.08

    # 运行采样
    pts = poisson_disk_sampling_3d(W, H, D, R, seed=42)
    print("生成的点数:", len(pts))

    # 画图
    try:
        import matplotlib.pyplot as plt

        # 需要 3D 绘图支持，必须引用 Axes3D
        from mpl_toolkits.mplot3d import Axes3D

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, s=5)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_zlim(0, D)
        ax.set_title(f"Poisson Disk Sampling 3D (N={len(pts)})")
        plt.show()

    except ImportError:
        print("没有安装 matplotlib，无法画图，但计算已完成。")
