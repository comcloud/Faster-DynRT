import torch
import torch.nn as nn


class TraditionalAttentionLayer(nn.Module):
    def __init__(self, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim):
        super(TraditionalAttentionLayer, self).__init__()
        self.text_seq_len = text_seq_len
        self.image_block_num = image_block_num

        self.text_attention = nn.MultiheadAttention(embed_dim=text_hidden_dim, num_heads=1)
        self.image_attention = nn.MultiheadAttention(embed_dim=image_hidden_dim, num_heads=1)

        # 定义文本长度和图片块数互换
        self.seq_len_to_block = nn.Linear(text_seq_len, image_block_num)
        self.seq_len_from_block = nn.Linear(image_block_num, text_seq_len)
        # 定义图片块数和文本长度互换
        self.block_to_seq_len = nn.Linear(image_block_num, text_seq_len)
        self.block_from_seq_len = nn.Linear(text_seq_len, image_block_num)

    def forward(self, text_feature, image_feature):
        '''
        :param text_feature: 文本特征，(batch_size, text_seq_len, text_hidden_dim)
        :param image_feature: 图片特征，(batch_size, image_block_num, image_hidden_dim)
        '''
        q_text = self.seq_len_to_block(text_feature.transpose(2, 1)).transpose(2, 1)
        q_image = self.block_to_seq_len(image_feature.transpose(2, 1)).transpose(2, 1)

        output_text, _ = self.text_attention(q_text, image_feature, image_feature)
        output_image, _ = self.image_attention(q_image, text_feature, text_feature)

        output_text = self.seq_len_from_block(output_text.transpose(2, 1)).transpose(2, 1)
        output_image = self.block_from_seq_len(output_image.transpose(2, 1)).transpose(2, 1)

        return output_text, output_image


if __name__ == '__main__':
    text_input = torch.randn(8, 128, 768)  # 输入张量大小为 (batch_size, seq_len, hidden_dim)
    image_input = torch.randn(8, 197, 768)  # 输入张量大小为 (batch_size, seq_len, hidden_dim)
    attr = TraditionalAttentionLayer(128, 768, 197, 768)
    res1, res2 = attr(text_input, image_input)
    print(res1.size())  # 输出张量大小为 (batch_size, seq_len, hidden_dim)
    print(res2.size())  # 输出张量大小为 (batch_size, seq_len, hidden_dim)
