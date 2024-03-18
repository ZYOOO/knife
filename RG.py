# -*- coding: utf-8 -*-            
# @Author : zhangyong
# @Time : 2024/1/17 上午12:36
import cv2
import numpy as np

# 读取图像
img = cv2.imread('imgs2/1-500.png', 0)  # 以灰度模式读取图像

# 定义种子点（可以手动选择或根据需求自动选择）
seed_point = (400, 360)  # 以(100, 100)为种子点

# 定义阈值，表示相似性的条件
threshold = 20

# 创建一个与图像相同大小的标记图像，用于记录分割结果
height, width = img.shape[:2]
visited = np.zeros((height, width), dtype=np.uint8)

# 定义一个队列来执行区域生长
queue = []

# 获取种子点的灰度值
seed_value = img[seed_point]

# 定义相邻像素的8个连接方式
neighbors = [(0, -1), (-1, 0), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# 区域生长算法
def region_growing(seed, threshold):
    queue.append(seed)
    visited[seed] = 255

    while len(queue) > 0:
        current_point = queue.pop(0)

        # 检查相邻像素
        for neighbor in neighbors:
            x, y = current_point[0] + neighbor[0], current_point[1] + neighbor[1]

            # 检查是否在图像范围内
            if 0 <= x < height and 0 <= y < width:
                # 检查相似性条件
                if abs(int(img[x, y]) - int(seed_value)) < threshold and visited[x, y] == 0:
                    queue.append((x, y))
                    visited[x, y] = 255

# 执行区域生长
region_growing(seed_point, threshold)

# 将分割结果保存为图像
result_image = cv2.merge([visited, visited, visited])  # 转换为3通道图像
# cv2.imwrite('segmented_image.jpg', result_image)
cv2.imshow('rg', result_image)
cv2.waitKey(0)