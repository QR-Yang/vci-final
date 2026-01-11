import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # 导入 PIL 用于处理图片
from sample_2D import poisson_disk_sampling_2d

def load_image_gray(filepath, width, height):
    """
    加载图片并转换为灰度图，同时缩放到指定大小
    """
    try:
        img = Image.open(filepath).convert('L')  # 转灰度
        img = img.resize((width, height))
        img_array = np.array(img).astype(float) / 255.0  # 归一化到 0-1
        return img_array
    except FileNotFoundError:
        print(f"Error: 找不到图片 {filepath}")
        # 如果找不到图片，回退到渐变图
        return generate_gradient_image(width, height)

def generate_gradient_image(width, height):
    """
    生成一个从黑到白的灰度渐变图 (0.0 到 1.0)
    """
    # 创建一个 meshgrid
    # x 从 0 到 1
    x = np.linspace(0, 1, width)
    # y 从 0 到 1
    y = np.linspace(0, 1, height)
    # 广播生成 2D 数组，这里只让它沿 X 轴渐变
    xv, yv = np.meshgrid(x, y)
    return xv

def create_blue_noise_noise_map(width, height, r):
    """
    利用泊松盘采样生成一个蓝噪声“扰动图”。
    虽然泊松盘采样生成的是离散的点，但我们可以把这些点看作是
    噪声的正峰值位置。
    """
    print(f"正在生成蓝噪声采样 (W={width}, H={height}, r={r})...")
    t0 = time.time()
    
    # 调用之前的算法生成点
    # 注意：r 是相对于 width/height 的浮点数
    # 为了保证有点密集度，r 应该比较小
    points = poisson_disk_sampling_2d(float(width), float(height), r, k=30)
    
    print(f"采样完成，耗时 {time.time() - t0:.4f}s，生成了 {len(points)} 个点")
    
    # 1. 创建一个全 0 的网格
    noise_grid = np.zeros((height, width), dtype=float)
    
    # 2. 把生成的点的位置设为 1.0
    for px, py in points:
        ix = int(px)
        iy = int(py)
        if 0 <= ix < width and 0 <= iy < height:
            noise_grid[iy, ix] = 1.0
            
    # 3. 简单的平滑/模糊处理，让噪声连续化（模拟蓝噪声纹理）
    # 这里用一个简单的高斯核或者平均核
    # 为了简单不引入 scipy，我们手动做个简单的卷积或者由于点很密集直接用
    # 这里我们直接把二值噪声减去均值作为扰动
    # 更好的做法是稍微模糊一下，让能量扩散
    
    # 简单 3x3 模糊
    kernel = np.array([[1, 2, 1], 
                       [2, 4, 2], 
                       [1, 2, 1]]) / 16.0
    
    # 利用 scipy 如果有的话，没有就跳过模糊直接用二值（效果会硬一点）
    try:
        from scipy.ndimage import convolve
        noise_grid = convolve(noise_grid, kernel)
        # 再做一次让它更平滑
        noise_grid = convolve(noise_grid, kernel)
        noise_grid = convolve(noise_grid, kernel)
    except ImportError:
        print("未检测到 scipy，跳过平滑步骤，使用原始二值噪声")
        pass

    # 4. 归一化到 [-0.5, 0.5] 或者其他范围
    # 也就是让噪声有正有负
    mean_val = np.mean(noise_grid)
    max_val = np.max(noise_grid) if np.max(noise_grid) > 0 else 1.0
    min_val = np.min(noise_grid)
    
    # 归一化到 -1 到 1
    if max_val - min_val > 0:
        noise_grid = (noise_grid - min_val) / (max_val - min_val) # 0~1
        noise_grid = noise_grid - 0.5 # -0.5 ~ 0.5
        noise_grid = noise_grid * 2.0 # -1.0 ~ 1.0
        
    return noise_grid

def dither_pipeline(image_path="david.jpg", width=400, height=400): # 增加分辨率以便看清图片细节
    # 1. 准备原图 (加载 david.jpg)
    print(f"加载图片: {image_path}...")
    img_gray = load_image_gray(image_path, width, height)
    
    # 2. 生成白噪声 (完全随机) 用于对比
    # 同样归一化到 -1 到 1
    white_noise = np.random.rand(height, width) * 2.0 - 1.0
    
    # 3. 生成蓝噪声 (基于泊松盘采样)
    # r 选小一点，比如 2.0 像素，这样点比较密，类似噪声纹理
    blue_noise = create_blue_noise_noise_map(width, height, r=1.2)
    
    # 4. 设定阈值算法
    # 核心公式: Output = 1 if (Pixel + Noise * Strength) > 0.5 else 0
    strength = 0.5 # 对于真实图片，稍微降低一点强度可能保留更多细节，也可以尝试 0.5
    
    # 白噪声抖动
    wb_dithering = img_gray + white_noise * strength
    wb_result = (wb_dithering > 0.5).astype(float)
    
    # 蓝噪声抖动
    bn_dithering = img_gray + blue_noise * strength
    bn_result = (bn_dithering > 0.5).astype(float)
    
    # 5. 可视化对比
    plt.figure(figsize=(12, 8))
    
    # 原图
    plt.subplot(2, 2, 1)
    plt.title("Original Image (Gray)")
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    # 纯阈值 (没有噪声)
    plt.subplot(2, 2, 2)
    plt.title("Simple Threshold (No Noise)")
    no_noise_result = (img_gray > 0.5).astype(float)
    plt.imshow(no_noise_result, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    # 白噪声抖动结果
    plt.subplot(2, 2, 3)
    plt.title("White Noise Dithering")
    plt.imshow(wb_result, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    # 蓝噪声抖动结果
    plt.subplot(2, 2, 4)
    plt.title("Blue Noise Dithering (Our Algorithm)")
    plt.imshow(bn_result, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dither_pipeline()  # 默认使用 david.jpg
