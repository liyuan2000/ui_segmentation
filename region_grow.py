import cv2
import numpy as np
from collections import deque


def region_growing(image, seed, threshold=10):
    height, width = image.shape
    segmented_image = np.zeros_like(image)
    visited = np.zeros_like(image, dtype=bool)

    seed_value = image[seed[1], seed[0]]
    queue = deque([seed])

    while queue:
        x, y = queue.popleft()
        if visited[y, x]:
            continue

        visited[y, x] = True
        segmented_image[y, x] = 255

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                if abs(int(image[ny, nx]) - int(seed_value)) < threshold:
                    queue.append((nx, ny))

    return segmented_image


# 读取图像
image = cv2.imread(r"D:\job\ui1.png", cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功加载
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# 定义种子点，格式为(x, y)
seed_point = (1, 1)

# 应用区域增长法
segmented_image = region_growing(image, seed_point, threshold=10)

# 显示原始图像和分割后的图像
# cv2.imwrite('Original Image', image)
cv2.imwrite('region_grow.png', segmented_image)
