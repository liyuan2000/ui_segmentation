import cv2
import numpy as np

# 读取图像
image = cv2.imread(r"D:\job\ui2.png")
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# 创建一个掩码图像
mask = np.zeros(image.shape[:2], np.uint8)

# 创建一个矩形，该矩形包含前景对象
rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

# 创建临时数组，用于grabCut函数的内部计算
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# 应用GrabCut算法
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# 将确定背景和可能背景的像素设置为0，将确定前景和可能前景的像素设置为1
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# 使用掩码提取前景
segmented_image = image * mask2[:, :, np.newaxis]

# 显示原始图像和分割后的图像
cv2.imwrite('graph.png', segmented_image)