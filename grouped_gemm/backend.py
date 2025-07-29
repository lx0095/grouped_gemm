# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

def gmm(a: torch.Tensor, 
        b: torch.Tensor, 
        batch_sizes: torch.Tensor, 
        trans_a: bool = False, 
        trans_b: bool = False) -> torch.Tensor:

    device = a.device
    num_experts = batch_sizes.numel()
    
    # 计算累积和，用于拆分输入张量
    batch_cumsum = torch.cat([torch.tensor([0], device=device), batch_sizes.cumsum(dim=0)])
    
    # 存储所有分组的计算结果
    results = []
    
    for i in range(num_experts):
        # 提取当前分组的尺寸
        size_i = batch_sizes[i].item()
        if size_i == 0:
            # 处理k=0的特殊情况（结果为0）
            if trans_a:
                m = a.size(1)
                n = b.size(1) if not trans_b else b.size(0)
                results.append(torch.zeros(m, n, device=device, dtype=a.dtype))
            else:
                n = b.size(1) if not trans_b else b.size(0)
                results.append(torch.zeros(size_i, n, device=device, dtype=a.dtype))
            continue
        
        # 拆分A的第i个分组
        a_start = batch_cumsum[i]
        a_end = batch_cumsum[i+1]
        a_i = a[a_start:a_end]
        if trans_a:
            a_i = a_i.t()  # 转置A_i
        
        # 拆分B的第i个分组（根据B的形状判断是否为3D批量格式）
        if b.dim() == 3:
            b_i = b[i]  # 3D格式: (num_experts, k, n) 或 (num_experts, n, k)
        else:
            b_start = batch_cumsum[i]
            b_end = batch_cumsum[i+1]
            b_i = b[b_start:b_end]
        if trans_b:
            b_i = b_i.t()  # 转置B_i
        
        # 执行当前分组的矩阵乘法: C_i = A_i @ B_i
        c_i = torch.matmul(a_i, b_i)
        results.append(c_i)
    
    # 拼接所有分组的结果（根据维度自动适配）
    return torch.cat(results, dim=0)


def _allocate_output(a, b, batch_sizes, trans_a, trans_b):
    assert not (trans_a and trans_b)
    assert batch_sizes.ndim == 1, "Expected 1d tensor for batch_sizes"
    assert a.ndim == 2, "Expected 2d tensor for 'a'"
    assert b.ndim == (2 if trans_a else 3)

    shape = (
        (batch_sizes.shape[0], a.shape[1], b.shape[1])
        if trans_a else
        (a.shape[0], (b.shape[1] if trans_b else b.shape[2]))
    )
    return torch.empty(*shape, device=a.device, dtype=a.dtype)
