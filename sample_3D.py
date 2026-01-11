import math
import random
def poisson_disk_sampling_3d(width, height, depth, r, k=30):
    cell_size = r / math.sqrt(3)
    cols = int(math.ceil(width / cell_size))
    rows = int(math.ceil(height / cell_size))
    layers = int(math.ceil(depth / cell_size))
    grid = [-1] * (cols * rows * layers)
    samples = []
    active_list = []
    x = random.random() * width
    y = random.random() * height
    z = random.random() * depth
    first_point = (x, y, z)
    samples.append(first_point)
    active_list.append(0)
    col = int(x / cell_size)
    row = int(y / cell_size)
    layer = int(z / cell_size)
    grid[(layer * rows + row) * cols + col] = 0
    while len(active_list) > 0:
        rand_index = random.randint(0, len(active_list) - 1)
        point_index = active_list[rand_index]
        px, py, pz = samples[point_index]
        found_new_point = False
        for i in range(k):
            dx = random.gauss(0, 1)
            dy = random.gauss(0, 1)
            dz = random.gauss(0, 1)
            # 这里取的是高斯分布的随机数
            d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d_len == 0:
                continue
            dx /= d_len
            dy /= d_len
            dz /= d_len
            u = random.random()
            radius = r * (1 + 7 * u) ** (1 / 3)
            #这里是体积所以取三分之一次方
            new_x = px + dx * radius
            new_y = py + dy * radius
            new_z = pz + dz * radius
            if not (0 <= new_x < width and 0 <= new_y < height and 0 <= new_z < depth):continue
            new_col = int(new_x / cell_size)
            new_row = int(new_y / cell_size)
            new_layer = int(new_z / cell_size)
            too_close = False
            r_sq = r * r
            layer_min = max(0, new_layer - 2)
            layer_max = min(layers, new_layer + 3)
            row_min = max(0, new_row - 2)
            row_max = min(rows, new_row + 3)
            col_min = max(0, new_col - 2)
            col_max = min(cols, new_col + 3)
            for l_idx in range(layer_min, layer_max):
                for r_idx in range(row_min, row_max):
                    for c_idx in range(col_min, col_max):
                        neighbor_idx = grid[(l_idx * rows + r_idx) * cols + c_idx]
                        if neighbor_idx != -1:
                            ex, ey, ez = samples[neighbor_idx]
                            dist_sq = (ex - new_x) ** 2 + (ey - new_y) ** 2 + (ez - new_z) ** 2
                            if dist_sq < r_sq:
                                too_close = True
                                break
                    if too_close:
                        break
                if too_close:
                    break
            if not too_close:
                found_new_point = True
                samples.append((new_x, new_y, new_z))
                new_idx = len(samples) - 1
                active_list.append(new_idx)
                grid[(new_layer * rows + new_row) * cols + new_col] = new_idx
                break
        if not found_new_point: active_list.pop(rand_index)
    return samples
if __name__ == "__main__":
    W, H, D = 1.0, 1.0, 1.0
    R = 0.08
    pts = poisson_disk_sampling_3d(W, H, D, R)
    print("生成的点数:", len(pts))
    import matplotlib.pyplot as plt
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
