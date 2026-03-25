import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_with_tsne(features, labels, selected_labels, out_path, rel_type, steps=0, method="crt"):

    """
    使用T-SNE对特征进行降维，并可视化结果。

    Args:
    - features (np.array): 特征数组，形状为 (num_samples, feature_dim)。
    - labels (np.array): 标签数组，形状为 (num_samples,)。
    - num_classes (int): 要可视化的类别数量。
    """
    save_path = os.path.join(out_path, 'stabile_' + rel_type + '.png')
    # 初始化T-SNE对象
    tsne = TSNE(n_components=2, random_state=0)

    # 对特征进行降维
    tsne_results = tsne.fit_transform(features)

    # 为每个类别选择一个颜色
    palette = np.array(plt.cm.tab10.colors)

    # 创建一个图来可视化数据
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(selected_labels):
        # 绘制每个选定类别的数据点
        plt.scatter(tsne_results[labels == label, 0], tsne_results[labels == label, 1], c=[palette[i % len(palette)]], label=f"Class {label}")
        # 计算并绘制每个类别的原型
        # prototype = np.mean(tsne_results[labels == label, :], axis=0)
        # plt.scatter(prototype[0], prototype[1], c=[palette[i % len(palette)]], edgecolor='k', marker='*', s=300, label=f"Prototype {label}")


    # plt.legend(loc='best', fontsize='large')
    # plt.xlabel('t-SNE feature 0')
    # plt.ylabel('t-SNE feature 1')
    # plt.title('T-SNE Visualization of the Features')
    plt.savefig(save_path, dpi=600)
    # plt.show()
