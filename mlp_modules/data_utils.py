import numpy as np
import pickle
import os
from sklearn.preprocessing import OneHotEncoder

def load_CIFAR10_data(data_dir):
    """加载 CIFAR-10 数据集并进行预处理。"""
    def unpickle(file):
        with open(file, 'rb') as f:
            dict_data = pickle.load(f, encoding='latin1')
        return dict_data

    # 读取训练数据
    X_train = []
    y_train = []
    for batch in range(1, 6):
        data_dict = unpickle(os.path.join(data_dir, f"data_batch_{batch}"))
        X_train.append(data_dict['data'])
        y_train.append(data_dict['labels'])
    
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # 读取测试数据
    data_dict = unpickle(os.path.join(data_dir, 'test_batch'))
    X_test = data_dict['data']
    y_test = np.array(data_dict['labels'])

    # 归一化到 [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # 将形状调整为 (N, 3, 32, 32)
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)

    # Flatten 成 (N, 3072)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # One-Hot 编码
    try:
        # 尝试新版本 scikit-learn 的参数
        encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        # 尝试旧版本 scikit-learn 的参数
        encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    return X_train, y_train, X_test, y_test

def split_train_val(X, y, val_ratio=0.2):
    """将训练数据分割为训练集和验证集"""
    n_samples = X.shape[0]
    n_val = int(n_samples * val_ratio)
    
    # 随机打乱数据
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    return X_train, y_train, X_val, y_val

def random_flip(X, p=0.5):
    """随机水平翻转图像，用于数据增强"""
    # X shape: (batch_size, 3072)
    batch_size = X.shape[0]
    X_reshaped = X.reshape(batch_size, 3, 32, 32)
    
    # 随机选择需要翻转的图像
    flip_mask = np.random.random(batch_size) < p
    
    # 水平翻转被选中的图像
    X_reshaped[flip_mask] = X_reshaped[flip_mask, :, :, ::-1]
    
    # 恢复原始形状
    return X_reshaped.reshape(batch_size, -1) 