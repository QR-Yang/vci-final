import time
import random
import matplotlib.pyplot as plt
from sample_2D import poisson_disk_sampling_2d
def benchmark_efficiency():
    print("\n--- 算法效率基准测试 ---")
    width, height = 1.0, 1.0
    test_radii = [0.1, 0.05, 0.025, 0.015] 
    for r in test_radii:
        start_time = time.time()
        points = poisson_disk_sampling_2d(width, height, r, k=30)
        end_time = time.time()
        duration = end_time - start_time
        count = len(points)
        speed = duration / count * 1000
        print(f"{r:<10.3f} | {count:<10d} | {duration:<15.4f} | {speed:.10f} ms/point")
def simulate_forest_generation():
    print("\n--- 下游应用演示：程序化森林生成 ---")
    width, height = 1.0, 1.0
    r = 0.05
    forest_poisson = poisson_disk_sampling_2d(width, height, r)
    num_trees = len(forest_poisson)
    forest_random = []
    for i in range(num_trees):
        forest_random.append((random.random() * width, random.random() * height))
    print(f"生成了 {num_trees} 棵树")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    def plot_stumps(ax, points, title, color):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=10, c='brown', label='Trunk')
        for x, y in points:
            circle = plt.Circle((x, y), r/2, color=color, alpha=0.3)
            ax.add_patch(circle)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_title(title)
    plot_stumps(axes[0], forest_random, f"Pure Random Generation", 'red')
    plot_stumps(axes[1], forest_poisson, f"Poisson Disk Generation", 'green')
    plt.show()
if __name__ == "__main__":
    benchmark_efficiency()
    simulate_forest_generation()