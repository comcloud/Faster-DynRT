import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['font.family'] = 'sans-serif'  # 用来正常显示中文标签


def load_file(filename):
    import pickle
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret


def text_attention_visualization(id, text=None, attention_weight=None):
    # 降维到单词个数
    len_words = len(text.split())
    k = len_words  # 设置降维后的维度
    tsne = TSNE(n_components=k, method='exact')
    # k, 100
    attention_weight = np.transpose(tsne.fit_transform(attention_weight.transpose(1, 0).detach().numpy()), axes=(1, 0))
    k = 1  # 设置降维后的维度
    tsne = TSNE(n_components=k, method='exact', perplexity=len_words - 1)
    attention_weight = np.transpose(tsne.fit_transform(attention_weight), axes=(1, 0))

    # 准备注释文本（即注意力权重）
    # annotations = np.round(attention_weight, decimals=2)  # 将注意力权重四舍五入到两位小数

    # 绘制文本序列和注意力权重对应关系
    plt.figure(figsize=(8, 2))
    sns.heatmap(attention_weight, annot=False, fmt='', cmap='Reds')
    plt.title('文本Attention可视化')
    plt.yticks([], [])
    plt.xticks(np.arange(len_words) + 0.5, [str(j + 1) for j in range(len_words)], rotation=0)
    plt.savefig('input/prepared/' + str(id) + '_txt.jpg', dpi=300)
    # plt.show()
    plt.close()
