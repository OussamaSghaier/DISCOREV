import torch


def softargmax(x, device, beta=100.0):
    """x: (batch_size, seq_len, voacb_size))"""
    b = torch.tensor([beta], device=device)
    xx = x * b
    sm = torch.nn.functional.softmax(xx, dim=-1)
    indices = torch.arange(x.size(-1),
                           device=device).repeat(x.size(0), x.size(1), 1)
    y = torch.mul(indices, sm)
    result = torch.sum(y, dim=-1)
    return result


def differentiable_round(x):
    return (x + 0.5).long()

