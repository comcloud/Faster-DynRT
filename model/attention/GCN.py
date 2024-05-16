import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x_batch):
        # x_batch 的形状为 (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x_batch.size()

        # 初始化一个空的 edge_index 列表
        batch_edge_index = []

        # 对每个样本
        for i in range(batch_size):
            # 获取当前样本的文本表示
            x = x_batch[i]  # 形状为 (seq_len, input_dim)
            # 调用 get_edge_index 方法获取边索引
            edge_index = self.get_edge_index(x)
            batch_edge_index.append(edge_index)

        # 对每个样本分别进行图卷积操作
        output_batch = []
        for x, edge_index in zip(x_batch, batch_edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            output_batch.append(x)

        # 将结果堆叠成一个张量
        output_batch = torch.stack(output_batch, dim=0)

        return output_batch

    def get_edge_index(self, x):
        # x 的形状为 (seq_len, input_dim)
        seq_len = x.size(0)
        # 计算每个节点与其最近的 k 个邻居之间的边
        k = 5
        distances = torch.cdist(x, x)
        _, knn_indices = torch.topk(distances, k + 1, dim=-1, largest=False)

        # 构造 edge_index
        edge_index = []
        for j in range(seq_len):
            source_node = j
            for neighbor_index in knn_indices[j]:
                if neighbor_index != j:
                    target_node = neighbor_index
                    edge_index.append([source_node, target_node])

        # 转换为 PyTorch 张量并返回
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device=device)
