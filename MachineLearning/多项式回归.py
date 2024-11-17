# --coding:utf-8--
"""
多项式回归
"""
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp
import sklearn.metrics as sm
import sklearn.preprocessing as sp
import sklearn.pipeline as pl
import pandas as pd
import joblib

def find_and_compute_areas(matrix, unit_area=0.8):
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    m, n = matrix.shape
    visited = np.zeros_like(matrix, dtype=bool)
    regions = []

    def dfs(i, j, label):
        if i < 0 or i >= m or j < 0 or j >= n or matrix[i, j] == 0 or visited[i, j]:
            return 0

        visited[i, j] = True

        if len(regions) <= label:
            regions.append(set())

        regions[label].add((i, j))

        count = 1  # 当前元素是1，因此从这个元素开始计数
        # 递归搜索相邻的1
        count += dfs(i - 1, j, label)
        count += dfs(i + 1, j, label)
        count += dfs(i, j - 1, label)
        count += dfs(i, j + 1, label)

        return count

    label = 0
    areas = []
    for i in range(m):
        for j in range(n):
            if matrix[i, j] == 1 and not visited[i, j]:
                count = dfs(i, j, label)
                area = count * unit_area
                areas.append(area)
                label += 1

    return areas

# 示例用法
matrix = np.array([
    [1, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 2, 0, 1, 1, 0],
    [0, 0, 0, 2, 2, 0, 0, 0, 0]
])

result = find_and_compute_areas(matrix, unit_area=0.8)
print(result)