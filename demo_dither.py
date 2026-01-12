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
def generate_gradient_image(width, height):
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    return xv
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
    mean_val = np.mean(noise_grid)
    max_val = np.max(noise_grid) if np.max(noise_grid) > 0 else 1.0
    min_val = np.min(noise_grid)
    if max_val - min_val > 0:
        noise_grid = (noise_grid - min_val) / (max_val - min_val)
        noise_grid = noise_grid - 0.5
        noise_grid = noise_grid * 2.0
    return noise_grid
def dither_pipeline(image_path="david.jpg", width=400, height=400):
    img_gray = load_image_gray(image_path, width, height)
    blue_noise = create_blue_noise_noise_map(width, height, r=1.2)
    strength = 0.5
    bn_dithering = img_gray + blue_noise * strength
    bn_result = (bn_dithering > 0.5).astype(float)
    plt.figure(figsize=(8, 5))
    plt.subplot(2, 2, 1)
    plt.title("Original Image (Gray)")
    plt.imshow(img_gray, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.title("Blue Noise Dithering (Our Algorithm)")
    plt.imshow(bn_result, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')  
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    dither_pipeline()