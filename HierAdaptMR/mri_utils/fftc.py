import torch
import torch.fft

def fft2c(data):
    """
    带中心平移的 2D 快速傅里叶变换 (Centric FFT)
    输入 data: [..., H, W, 2] (最后一位是实部虚部)
    """
    # 将最后一位 [real, imag] 转为复数张量
    data_complex = torch.view_as_complex(data)
    
    # 中心平移 -> FFT -> 中心平移
    data_complex = torch.fft.ifftshift(data_complex, dim=(-2, -1))
    data_complex = torch.fft.fft2(data_complex, dim=(-2, -1), norm='ortho')
    data_complex = torch.fft.fftshift(data_complex, dim=(-2, -1))
    
    # 转回 [..., H, W, 2] 格式
    return torch.view_as_real(data_complex)

def ifft2c(data):
    """
    带中心平移的 2D 逆快速傅里叶变换 (Centric IFFT)
    """
    data_complex = torch.view_as_complex(data)
    
    data_complex = torch.fft.ifftshift(data_complex, dim=(-2, -1))
    data_complex = torch.fft.ifft2(data_complex, dim=(-2, -1), norm='ortho')
    data_complex = torch.fft.fftshift(data_complex, dim=(-2, -1))
    
    return torch.view_as_real(data_complex)
