import torch

def complex_abs(data):
    assert data.shape[-1] == 2
    return torch.sqrt(data.pow(2).sum(-1))

def complex_mul(t1, t2):
    assert t1.shape[-1] == 2
    assert t2.shape[-1] == 2
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    res_real = real1 * real2 - imag1 * imag2
    res_imag = real1 * imag2 + imag1 * real2
    return torch.stack((res_real, res_imag), dim=-1)

def complex_conj(data):
    assert data.shape[-1] == 2
    out = data.clone()
    out[..., 1] = -out[..., 1]
    return out
