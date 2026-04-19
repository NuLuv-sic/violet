import scipy.io
import h5py
import numpy as np
import torch

def to_tensor(data):
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)

def ifft2c(data):
    # 输入应为 [..., H, W, 2]
    data = torch.view_as_complex(data)
    data = torch.fft.ifftshift(data, dim=(-2, -1))
    data = torch.fft.ifft2(data, dim=(-2, -1), norm="ortho")
    data = torch.fft.fftshift(data, dim=(-2, -1))
    data = torch.view_as_real(data)
    return data

def rss_complex(data, dim=0):
    # Root Sum of Squares for complex data
    return torch.sqrt((data**2).sum(dim).sum(-1))

def load_kdata(file_path):
    try:
        # 尝试用 scipy 加载 (针对旧版 v7.2 mat)
        data = scipy.io.loadmat(file_path)
        # 寻找可能的 key (比赛数据通常叫 kspace 或 kdata)
        for key in ['kspace', 'kdata', 'KData', 'T1rho', 'cine_sax']:
            if key in data:
                return data[key]
        # 如果没找到特定的 key，返回第一个不是 __ 打头的变量
        for k, v in data.items():
            if not k.startswith('__'):
                return v
    except:
        # 如果 scipy 失败，尝试用 h5py 加载 (针对新版 v7.3 mat)
        with h5py.File(file_path, 'r') as f:
            for key in ['kspace', 'kdata', 'KData', 'T1rho', 'cine_sax']:
                if key in f:
                    return np.array(f[key])
            # 如果没找到，返回第一个 dataset
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    return np.array(f[k])
                    
    raise KeyError(f"Could not find valid data key in {file_path}")