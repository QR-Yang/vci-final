import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sample_2D import poisson_disk_sampling_2d
def load_image_gray(filepath, width, height):
    img = Image.open(filepath).convert('L')
    img = img.resize((width, height))
    img_array = np.array(img).astype(float) / 255.0
    return img_array
def create_blue_noise_noise_map(width, height, r):
    points = poisson_disk_sampling_2d(float(width), float(height), r, k=30)
    noise_grid = np.zeros((height, width), dtype=float)
    for px, py in points:
        ix = int(px)
        iy = int(py)
        if 0 <= ix < width and 0 <= iy < height:
            noise_grid[iy, ix] = 1.0
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
    from scipy.ndimage import convolve
    noise_grid = convolve(noise_grid, kernel)
    noise_grid = convolve(noise_grid, kernel)
    noise_grid = convolve(noise_grid, kernel)
    #通过高斯模糊来把单独的噪音点传递到相邻区域
    flat = noise_grid.flatten()
    ranks = flat.argsort().argsort()
    #得到原始数组中每个元素在从小到大排序中的排名,只计算相对值
    uniform_noise = ranks.astype(float) / (len(flat) - 1)
    noise_grid = uniform_noise.reshape((height, width))
    return noise_grid
def dither_pipeline(image_path="david.jpg", width=640, height=640):
    img_gray = load_image_gray(image_path, width, height)
    blue_noise = create_blue_noise_noise_map(width, height, r=0.9)
    bn_result = (img_gray > blue_noise).astype(float)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=1)
    plt.axis('off') 
    plt.subplot(1, 2, 2)
    plt.title("Blue Noise Dithering")
    plt.imshow(bn_result, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')  
    plt.tight_layout()
    plt.savefig("dither_result.png")
    plt.show()
if __name__ == "__main__":
    dither_pipeline()