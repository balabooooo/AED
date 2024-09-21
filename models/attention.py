import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qk_bias=True, attn_drop=0.0, extra_value=5.0,
                 width=200, dummy_value=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = float(head_dim) ** -0.5
        self.extra_value = extra_value
        self.width = width
        self.dummy_value = dummy_value

        self.linear = nn.Linear(dim, dim, bias=qk_bias)
        self.dropout = nn.Dropout(attn_drop)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.)

    def forward(self, query, key, mask=None):
        Q_B, Q_N, Q_C = query.shape
        K_B, K_N, K_C = key.shape
        assert Q_C == K_C and Q_B == K_B and Q_C == self.dim
        if Q_N == 0 or K_N == 0:
            return torch.empty([Q_B, Q_N, K_N]).to(query.device), torch.empty([Q_B, Q_N, K_N]).to(query.device)
        # q: B, num_heads, Q_N, C, head_dim
        q = self.linear(query).reshape(Q_B, Q_N, self.num_heads, Q_C // self.num_heads).permute(0, 2, 1, 3)   
        # k: B, num_heads, K_N, C, head_dim
        k = self.linear(key).reshape(K_B, K_N, self.num_heads, K_C // self.num_heads).permute(0, 2, 1, 3)

        weight_mm = (q @ k.transpose(-2, -1))

        weight_cos = F.cosine_similarity(k.unsqueeze(2), q.unsqueeze(3), dim=-1)

        if mask is not None:
            weight_cos = weight_cos.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)
            weight_mm = weight_mm.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        weight_cos = self.dropout(weight_cos)
        weight_mm = self.dropout(weight_mm)
        weight_cos = weight_cos.sum(dim=1) / self.num_heads
        weight_mm = weight_mm.sum(dim=1) / self.num_heads

        weight_cos = torch.clamp(weight_cos, min=0)
        return weight_cos, weight_mm


if __name__ == '__main__':
    # test
    # q: [B, N, C]
    # k: [B, N, C]
    dim = 256
    q = torch.randn(1, 2, dim)
    k = torch.randn(1, 3, dim)
    # mask: [B, N]
    attn = WeightAttention(dim)(q, k)
    print(attn.shape)  # [B, num_heads, N, N]
    print(attn)