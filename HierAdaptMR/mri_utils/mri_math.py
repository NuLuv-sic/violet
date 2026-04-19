import torch

def complex_abs(data):
    # 输入为 [..., 2]，最后一位是实部和虚部
    assert data.shape[-1] == 2
    return torch.sqrt(data.pow(2).sum(-1))

def complex_mul(t1, t2):
    # 复数乘法 (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    assert t1.shape[-1] == 2
    assert t2.shape[-1] == 2
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    res_real = real1 * real2 - imag1 * imag2
    res_imag = real1 * imag2 + imag1 * real2
    return torch.stack((res_real, res_imag), dim=-1)

def complex_conj(data):
    # 复数共轭 a+bi -> a-bi
    assert data.shape[-1] == 2
    out = data.clone()
    out[..., 1] = -out[..., 1]
    return out
