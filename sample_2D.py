import math
import random
def poisson_disk_sampling_2d(width, height, r, k=30):
    cell_size = r / math.sqrt(2)
    cols = int(math.ceil(width / cell_size))
    rows = int(math.ceil(height / cell_size))
    grid = [-1] * (cols * rows)
    samples = []
    active_list = []
    x = random.random() * width
    y = random.random() * height
    first_point = (x, y)
    samples.append(first_point)
    active_list.append(0) 
    col = int(x / cell_size)
    row = int(y / cell_size)
    grid[row * cols + col] = 0
    while len(active_list) > 0:
        rand_index = random.randint(0, len(active_list) - 1)
        point_index = active_list[rand_index]
        px, py = samples[point_index]
        found_new_point = False
        for i in range(k):
            angle = 2 * math.pi * random.random()
            radius = r * math.sqrt(random.uniform(1, 4))
            # 这里我用的是在面积上随机采样,再开根回半径,不然会导致边缘点稀疏
            new_x = px + radius * math.cos(angle)
            new_y = py + radius * math.sin(angle)
            if not (0 <= new_x < width and 0 <= new_y < height): continue            
            new_col = int(new_x / cell_size)
            new_row = int(new_y / cell_size)
            too_close = False
            r_sq = r * r
            row_min = max(0, new_row - 2)
            row_max = min(rows, new_row + 3)
            col_min = max(0, new_col - 2)
            col_max = min(cols, new_col + 3)
            for r_idx in range(row_min, row_max):
                for c_idx in range(col_min, col_max):
                    existing_point_index = grid[r_idx * cols + c_idx]
                    if existing_point_index != -1:
                        ex, ey = samples[existing_point_index]
                        dist_sq = (ex - new_x)**2 + (ey - new_y)**2
                        if dist_sq < r_sq:
                            too_close = True
                            break
                if too_close:
                    break   
            if not too_close:
                found_new_point = True
                samples.append((new_x, new_y))
                new_idx = len(samples) - 1
                active_list.append(new_idx)
                grid[new_row * cols + new_col] = new_idx
                break
        if not found_new_point:
            active_list.pop(rand_index)
    return samples
if __name__ == "__main__":
    W, H = 1.0, 1.0
    R = 0.03
    pts = poisson_disk_sampling_2d(W, H, R)
    import matplotlib.pyplot as plt
    x_list = [p[0] for p in pts]
    y_list = [p[1] for p in pts]
    plt.figure(figsize=(6, 6))
    plt.scatter(x_list, y_list, s=5)
    plt.xlim(0, W)
    plt.ylim(0, H)
    plt.gca().set_aspect('equal')
    plt.title(f"Poisson Disk Sampling 2D (N={len(pts)})")
    plt.savefig("poisson_disk_sampling_2D.png")
    plt.show()