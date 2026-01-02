from torch import nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size # 嵌入向量的维度（每个输入向量的大小）
        self.heads = heads # 注意头数量
        self.head_dim = embed_size // heads # 每个注意力头维度大小

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"
        # 线性变换层
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False) # V
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False) # K
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False) # Q
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # 计算完成后将所有头的输出拼接起来

    def forward(self, values, keys, query, mask): # (batch_size, seq_len, embed_size)
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        # 将张量分割成heads个头 (batch_size, len, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        # 线性变换
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        # 计算注意力分数 (batch_size, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])# n：批量大小（Batch Size）。q：查询序列的长度。k：键序列的长度。h：头的数量。d：每个头的维度。
        # 应用掩码
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))# 将掩码为 0 的位置的注意力分数设置为一个非常小的值（如 -1e20），这样在后续的 softmax 计算中，这些位置的权重会趋近于 0。
        # 注意力权重
        attention = torch.softmax(energy / (self.embed_size ** (0.5)), dim=3)
        #
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)





