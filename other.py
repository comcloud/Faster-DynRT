import torch
import torch.nn.functional as F


def sparse_attention(query, key, value, sparsity):
    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 获取最大注意力分数的索引
    topk_values, topk_indices = torch.topk(scores, k=sparsity, dim=-1)

    # 根据稀疏索引从value中选择相应的值
    sparse_value = torch.gather(value, -2, topk_indices.unsqueeze(-1).expand_as(value))

    # 归一化稀疏值
    sparse_attention_weights = F.softmax(topk_values, dim=-1)

    # 将稀疏权重乘以对应的稀疏值
    sparse_output = torch.sum(sparse_attention_weights.unsqueeze(-1) * sparse_value, dim=-2)

    return sparse_output, sparse_attention_weights


