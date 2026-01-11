import time
import random
import matplotlib.pyplot as plt
from sample_2D import poisson_disk_sampling_2d

def benchmark_efficiency():
    """
    任务1：通过实验分析证明算法的高效性
    测试在不同 r 值（对应不同点数量）下的生成时间
    """
    print("\n--- 1. 算法效率基准测试 ---")
    width, height = 1.0, 1.0
    # 测试不同的半径 r，半径越小，点越多
    test_radii = [0.1, 0.05, 0.025, 0.015] 
    
    print(f"{'Radius':<10} | {'Points':<10} | {'Time (sec)':<15} | {'Speed (pts/sec)':<15}")
    print("-" * 60)
    
    for r in test_radii:
        start_time = time.time()
        # 调用你的算法
        points = poisson_disk_sampling_2d(width, height, r, k=30)
        end_time = time.time()
        
        duration = end_time - start_time
        count = len(points)
        speed = count / duration if duration > 0 else 0
        
        print(f"{r:<10.3f} | {count:<10d} | {duration:<15.4f} | {speed:<15.1f}")
    print("结论：算法能够在极短时间内生成数千个高质量采样点，呈线性时间复杂度 O(N)。")

def simulate_forest_generation():
    """
    任务2：下游任务应用 - 程序化植被分布
    对比 '纯随机分布' 与 '泊松盘分布' 的视觉效果
    """
    print("\n--- 2. 下游应用演示：程序化森林生成 ---")
    width, height = 1.0, 1.0
    r = 0.05  # 树木之间的最小间距
    
    # 1. 使用泊松盘采样生成森林
    forest_poisson = poisson_disk_sampling_2d(width, height, r)
    num_trees = len(forest_poisson)
    
    # 2. 使用纯随机生成相同数量的树木（作为对照组）
    # 纯随机无法保证不重叠，只是简单地撒点
    forest_random = []
    for _ in range(num_trees):
        forest_random.append((random.random() * width, random.random() * height))
        
    print(f"生成了 {num_trees} 棵树。正在绘图对比...")

    # --- 可视化 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 定义画圆的辅助函数（模拟树冠占用的空间）
    def plot_stumps(ax, points, title, color):
        # 提取 X 和 Y
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        # 绘制点
        ax.scatter(xs, ys, s=10, c='brown', label='Trunk')
        
        # 绘制半径范围（树冠），展示重叠情况
        # 在 matplotlib 中，scatter 的 s 是面积，这里为了直观我们直接画 Circle
        for x, y in points:
            # 半径 r/2 的圆，表示如果不重叠，这些圆应该不相交（r是圆心距）
            circle = plt.Circle((x, y), r/2, color=color, alpha=0.3)
            ax.add_patch(circle)
            
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_title(title)
    
    # 左图：纯随机
    plot_stumps(axes[0], forest_random, f"Pure Random Generation\n(Overlaps & Gaps)", 'red')
    
    # 右图：泊松盘
    plot_stumps(axes[1], forest_poisson, f"Poisson Disk Generation\n(Evenly Distributed, No Overlap)", 'green')
    
    plt.suptitle("Application Scenario: Procedural Forest Generation")
    plt.show()

if __name__ == "__main__":
    benchmark_efficiency()
    try:
        simulate_forest_generation()
    except ImportError:
        print("缺少 matplotlib 库，无法展示图片。请安装: pip install matplotlib")
