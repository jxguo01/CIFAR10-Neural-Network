import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# 添加中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def visualize_weights(model_path, output_dir=None, cifar10_classes=None):
    """
    可视化神经网络模型的权重参数
    
    参数:
        model_path: 模型权重文件路径
        output_dir: 可视化结果保存目录，如果为None则不保存
        cifar10_classes: CIFAR-10类别名称列表
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 如果没有提供类别名称，使用默认名称
    if cifar10_classes is None:
        cifar10_classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    # 加载模型权重
    print(f"加载模型权重: {model_path}")
    weights = np.load(model_path)
    
    # 提取权重矩阵
    W1 = weights['W1']  # 输入到隐藏层的权重
    b1 = weights['b1']  # 隐藏层的偏置
    W2 = weights['W2']  # 隐藏层到输出层的权重
    b2 = weights['b2']  # 输出层的偏置
    
    # 检查是否有批归一化参数
    has_batchnorm = 'gamma1' in weights
    if has_batchnorm:
        gamma1 = weights['gamma1']
        beta1 = weights['beta1']
        print("模型包含批归一化参数")
    
    # 打印权重形状信息
    print(f"W1 shape: {W1.shape}")
    print(f"b1 shape: {b1.shape}")
    print(f"W2 shape: {W2.shape}")
    print(f"b2 shape: {b2.shape}")
    
    # 可视化 W1 (输入到隐藏层的权重)
    visualize_W1(W1, output_dir)
    
    # 可视化 W2 (隐藏层到输出层的权重)
    visualize_W2(W2, cifar10_classes, output_dir)
    
    # 可视化批归一化参数
    if has_batchnorm:
        visualize_batchnorm(gamma1, beta1, output_dir)
    
    # 可视化权重统计信息
    visualize_weight_stats(W1, W2, output_dir)
    
    # 可视化重要特征
    visualize_important_features(W1, W2, cifar10_classes, output_dir)
    
    print("权重可视化完成。")

def visualize_W1(W1, output_dir=None):
    """可视化第一层权重矩阵（输入到隐藏层）"""
    # 创建一个大图
    plt.figure(figsize=(20, 12))
    plt.suptitle("隐藏层神经元滤波器", fontsize=16)
    
    # 确定要显示的神经元数量（最多36个）
    n_neurons = min(36, W1.shape[1])
    
    # 计算每行每列显示的子图数量
    n_rows = int(np.ceil(np.sqrt(n_neurons)))
    n_cols = int(np.ceil(n_neurons / n_rows))
    
    # 对每个神经元权重范数进行排序，选择前n_neurons个最大的
    weight_norms = np.linalg.norm(W1, axis=0)
    top_neurons = np.argsort(weight_norms)[-n_neurons:][::-1]
    
    # 遍历选定的神经元
    for i, neuron_idx in enumerate(top_neurons):
        weights = W1[:, neuron_idx]
        
        # 重塑为 3x32x32 (RGB图像形状)
        w_img = weights.reshape(3, 32, 32)
        
        # 转置为 32x32x3 便于matplotlib显示
        w_img = w_img.transpose(1, 2, 0)
        
        # 归一化到 [0, 1] 范围
        w_min, w_max = w_img.min(), w_img.max()
        w_img = (w_img - w_min) / (w_max - w_min + 1e-10)
        
        # 显示图像
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(w_img)
        plt.title(f"神经元 #{neuron_idx}\n范数: {weight_norms[neuron_idx]:.2f}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'W1_filters.png'), dpi=150)
        
    plt.show()
    plt.close()
    
    # 可视化权重分布
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(W1.flatten(), bins=50, alpha=0.7)
    plt.title('W1权重分布')
    plt.xlabel('权重值')
    plt.ylabel('频次')
    
    plt.subplot(1, 2, 2)
    plt.hist(weight_norms, bins=30, alpha=0.7)
    plt.title('W1神经元范数分布')
    plt.xlabel('范数值')
    plt.ylabel('频次')
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'W1_distribution.png'), dpi=150)
        
    plt.show()
    plt.close()

def visualize_W2(W2, class_names, output_dir=None):
    """可视化第二层权重矩阵（隐藏层到输出层）"""
    plt.figure(figsize=(15, 8))
    
    # 创建自定义的热力图颜色映射
    cmap = LinearSegmentedColormap.from_list('BlueRedCmap', ['blue', 'white', 'red'])
    
    # 热力图绘制
    im = plt.imshow(W2, cmap=cmap, aspect='auto')
    plt.colorbar(im, label='权重值')
    
    # 设置y轴标签为隐藏层单元编号
    if W2.shape[0] <= 20:  # 如果神经元数量少，显示每个
        plt.yticks(range(W2.shape[0]), [f'Hidden {i}' for i in range(W2.shape[0])])
    else:  # 否则只显示部分
        step = W2.shape[0] // 10
        plt.yticks(range(0, W2.shape[0], step), [f'Hidden {i}' for i in range(0, W2.shape[0], step)])
    
    # 设置x轴标签为类别名称
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    
    plt.title('隐藏层到输出层的权重连接')
    plt.xlabel('输出类别')
    plt.ylabel('隐藏层单元')
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'W2_heatmap.png'), dpi=150)
        
    plt.show()
    plt.close()
    
    # 计算每个输出类别的主要贡献神经元
    plt.figure(figsize=(15, 8))
    top_k = 5  # 每个类别显示前k个最重要的神经元
    
    # 为每个类别找到最重要的神经元
    for i, class_name in enumerate(class_names):
        weights = W2[:, i]
        # 获取前k个最大的权重及其索引
        top_indices = np.argsort(weights)[-top_k:][::-1]
        top_weights = weights[top_indices]
        
        plt.subplot(2, 5, i+1)
        plt.bar(range(top_k), top_weights)
        plt.title(f'{class_name}')
        plt.xlabel('Top Hidden Neurons')
        plt.ylabel('权重值')
        plt.xticks(range(top_k), top_indices)
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'W2_top_neurons.png'), dpi=150)
        
    plt.show()
    plt.close()

def visualize_batchnorm(gamma, beta, output_dir=None):
    """可视化批归一化参数"""
    plt.figure(figsize=(12, 6))
    
    # 创建坐标轴
    x = np.arange(len(gamma.flatten()))
    
    # 绘制gamma和beta
    plt.subplot(2, 1, 1)
    plt.bar(x, gamma.flatten(), alpha=0.7)
    plt.title('Batch Normalization Gamma (缩放因子)')
    plt.xlabel('神经元索引')
    plt.ylabel('Gamma值')
    
    plt.subplot(2, 1, 2)
    plt.bar(x, beta.flatten(), alpha=0.7)
    plt.title('Batch Normalization Beta (平移因子)')
    plt.xlabel('神经元索引')
    plt.ylabel('Beta值')
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'batchnorm_params.png'), dpi=150)
        
    plt.show()
    plt.close()
    
    # 可视化gamma和beta的分布
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.hist(gamma.flatten(), bins=30, alpha=0.7)
    plt.title('Gamma分布')
    plt.xlabel('Gamma值')
    plt.ylabel('频次')
    
    plt.subplot(2, 1, 2)
    plt.hist(beta.flatten(), bins=30, alpha=0.7)
    plt.title('Beta分布')
    plt.xlabel('Beta值')
    plt.ylabel('频次')
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'batchnorm_dist.png'), dpi=150)
        
    plt.show()
    plt.close()

def visualize_weight_stats(W1, W2, output_dir=None):
    """可视化权重的统计信息"""
    plt.figure(figsize=(12, 10))
    
    # 标题用于图表
    titles = ['W1 (输入->隐藏层)', 'W2 (隐藏层->输出层)']
    weights = [W1, W2]
    
    # 绘制权重分布直方图
    for i, (W, title) in enumerate(zip(weights, titles)):
        # 扁平化权重
        w_flat = W.flatten()
        
        # 基本统计信息
        mean = np.mean(w_flat)
        std = np.std(w_flat)
        median = np.median(w_flat)
        min_val = np.min(w_flat)
        max_val = np.max(w_flat)
        
        plt.subplot(2, 2, i+1)
        plt.hist(w_flat, bins=50, alpha=0.7)
        plt.title(f'{title} 权重分布')
        plt.xlabel('权重值')
        plt.ylabel('频次')
        
        # 添加垂直线标记均值和中位数
        plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'均值: {mean:.4f}')
        plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label=f'中位数: {median:.4f}')
        
        # 添加统计信息文本
        stat_text = f'均值: {mean:.4f}\n标准差: {std:.4f}\n中位数: {median:.4f}\n最小值: {min_val:.4f}\n最大值: {max_val:.4f}'
        plt.text(0.95, 0.95, stat_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 范数直方图
        norms = np.linalg.norm(W, axis=0)
        
        plt.subplot(2, 2, i+3)
        plt.hist(norms, bins=30, alpha=0.7)
        plt.title(f'{title} 神经元范数分布')
        plt.xlabel('范数值')
        plt.ylabel('频次')
        
        # 添加统计信息
        norm_mean = np.mean(norms)
        norm_std = np.std(norms)
        norm_text = f'均值: {norm_mean:.4f}\n标准差: {norm_std:.4f}'
        plt.text(0.95, 0.95, norm_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'weight_statistics.png'), dpi=150)
        
    plt.show()
    plt.close()

def visualize_important_features(W1, W2, class_names, output_dir=None):
    """为每个类别可视化重要特征"""
    # 图像大小
    img_size = (32, 32, 3)
    
    # 为每个类别计算重要特征
    plt.figure(figsize=(15, 10))
    
    for i, class_name in enumerate(class_names):
        # 获取该类别的输出权重
        class_weights = W2[:, i]
        
        # 计算综合权重：隐藏层单元对输入的权重乘以该单元对输出类别的权重
        # 形状变换：W1.shape = (3072, hidden_size), class_weights.shape = (hidden_size,)
        # 结果 combined.shape = (3072,)
        combined = W1.dot(class_weights)
        
        # 将权重重塑为图像形状
        feature_map = combined.reshape(img_size[2], img_size[0], img_size[1])
        # 转置为正确的图像格式
        feature_map = feature_map.transpose(1, 2, 0)
        
        # 归一化到 [0, 1] 范围，以便于可视化
        feature_map_abs = np.abs(feature_map)
        feature_map_norm = feature_map_abs / feature_map_abs.max()
        
        # 绘制热力图显示重要特征
        plt.subplot(2, 5, i+1)
        # 使用加权RGB通道
        rgb_img = feature_map_norm
        plt.imshow(rgb_img)
        plt.title(f'{class_name}的重要特征')
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'class_important_features.png'), dpi=150)
        
    plt.show()
    plt.close() 