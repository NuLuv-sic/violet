import os
import sys
import torch
import numpy as np

# 1. 强制注入项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 2. 模拟缺失的 mri_utils.math (这是你之前报错的根源)
from types import ModuleType
m_utils = ModuleType("mri_utils")
m_math = ModuleType("math")
def complex_abs(data): return torch.sqrt(data[..., 0]**2 + data[..., 1]**2)
def complex_mul(a, b):
    res_real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
    res_imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
    return torch.stack((res_real, res_imag), dim=-1)
def complex_conj(a):
    res = a.clone(); res[..., 1] = -res[..., 1]
    return res
m_math.complex_abs, m_math.complex_mul, m_math.complex_conj = complex_abs, complex_mul, complex_conj
m_utils.math = m_math
sys.modules["mri_utils"] = m_utils
sys.modules["mri_utils.math"] = m_math

# 3. 核心导入 (现在不会报错了)
from mri_network.multi_center_adapter import MultiCenterAdaptivePromptMR
from data_loading.data_module import FastMRIDataModule

def main():
    print("=== HierAdaptMR 启动检查 (4090 Cluster) ===")
    
    # 数据路径指向你刚才找到的目录
    DATA_ROOT = "/home/zb1/ChallengeData2025/ChallengeData/ChallengeData/MultiCoil"
    
    if not os.path.exists(DATA_ROOT):
        print(f"❌ 警告：找不到数据路径 {DATA_ROOT}")
        return

    # 初始化模型 (参数参考你的 settings.json)
    print("正在加载模型架构...")
    model = MultiCenterAdaptivePromptMR() 
    
    # 模拟一个 Batch 的数据流向
    print(f"✅ 模型加载成功！准备对接 {DATA_ROOT} 中的 .mat 数据...")
    print("提示：接下来需要根据导师给你的 .ckpt 文件进行权重加载。")

if __name__ == "__main__":
    main()