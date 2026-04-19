import torch
import numpy as np
import h5py

def load_shape(file_path):
    with h5py.File(file_path, 'r') as f:
        # 假设数据在 kspace 或 reconstruction 下
        if 'kspace' in f:
            return f['kspace'].shape
    return (0, 0, 0) # 兜底方案

def load_kdata(file_path):
    with h5py.File(file_path, 'r') as f:
        # 读取我们预处理生成的 h5 文件
        if 'kspace' in f:
            return np.array(f['kspace'])
    return None

def load_mask(file_path):
    with h5py.File(file_path, 'r') as f:
        if 'mask' in f:
            return np.array(f['mask'])
    # 如果没有 mask，返回全 1 的占位符（让代码能跑通）
    return np.ones((1, 1, 1))
