import numpy as np
import matplotlib.pyplot as plt
import os
from mlp_modules.data_utils import random_flip

def train_neural_network(model, X_train, y_train, X_val, y_val,
                        epochs=10, batch_size=64,
                        learning_rate=0.01, lambda_reg=0.01,
                        lr_decay=0.9, momentum=0.0, use_dropout=True,
                        early_stopping=10, save_dir='checkpoints'):
    """
    训练神经网络基础版
    
    参数:
        model: 神经网络模型
        X_train, y_train: 训练数据和标签
        X_val, y_val: 验证数据和标签
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 初始学习率
        lambda_reg: L2正则化系数
        lr_decay: 学习率衰减率
        momentum: 动量系数
        use_dropout: 是否使用dropout
        early_stopping: 早停轮数，0为不使用早停
        save_dir: 模型保存目录
        
    返回:
        best_val_acc: 最佳验证准确率
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    best_acc = 0.0
    best_weights = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # 打乱训练数据
        perm = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        # 按批次进行训练
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # 数据增强
            X_batch = random_flip(X_batch)
            
            # 前向传播（训练模式下使用dropout）
            model.forward(X_batch, is_training=use_dropout)
            # 反向传播并更新
            model.backward(X_batch, y_batch, learning_rate, lambda_reg, momentum)

        # 学习率衰减策略
        if epoch >= 5 and epoch % 10 == 0:
            learning_rate *= 0.5
        # 普通学习率衰减
        learning_rate *= lr_decay

        # 验证集准确率（不使用dropout）
        val_preds = model.forward(X_val, is_training=False)
        val_acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1))

        # 记录最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = (model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy())
            patience_counter = 0
            
            # 保存最佳模型
            if save_dir:
                model.save_weights(os.path.join(save_dir, 'best_model.npz'))
        else:
            patience_counter += 1
            
        # 早停
        if early_stopping > 0 and patience_counter >= early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.4f}")

    # 恢复最佳权重
    if best_weights is not None:
        model.W1, model.b1, model.W2, model.b2 = best_weights

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return best_acc


def train_neural_network_with_metrics(model, X_train, y_train, X_val, y_val,
                                     epochs=10, batch_size=64,
                                     learning_rate=0.01, lambda_reg=0.01,
                                     lr_decay=0.9, momentum=0.0, use_dropout=True,
                                     early_stopping=10, save_dir='checkpoints'):
    """
    训练神经网络并记录训练指标
    
    参数:
        model: 神经网络模型
        X_train, y_train: 训练数据和标签
        X_val, y_val: 验证数据和标签
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 初始学习率
        lambda_reg: L2正则化系数
        lr_decay: 学习率衰减率
        momentum: 动量系数
        use_dropout: 是否使用dropout
        early_stopping: 早停轮数，0为不使用早停
        save_dir: 模型保存目录
        
    返回:
        history: 包含训练历史的字典
        best_val_acc: 最佳验证准确率
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    best_acc = 0.0
    best_weights = None
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    orig_lr = learning_rate  # 保存初始学习率

    for epoch in range(epochs):
        # 打乱训练数据
        perm = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # 数据增强
            X_batch = random_flip(X_batch)

            # 前向传播
            preds = model.forward(X_batch, is_training=use_dropout)
            # 计算损失 (交叉熵)
            loss = -np.sum(y_batch * np.log(preds + 1e-10)) / X_batch.shape[0]
            # 准确率
            acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_batch, axis=1))

            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1

            # 反向传播
            model.backward(X_batch, y_batch, learning_rate, lambda_reg, momentum)

        # 学习率衰减 (step schedule)
        if epoch >= 5 and epoch % 10 == 0:
            learning_rate *= 0.5
        # 普通学习率衰减
        learning_rate *= lr_decay

        # 计算训练集平均损失和准确率
        train_losses.append(epoch_loss / num_batches)
        train_accuracies.append(epoch_acc / num_batches)

        # 验证集评估 (不使用dropout)
        val_preds = model.forward(X_val, is_training=False)
        val_loss = -np.sum(y_val * np.log(val_preds + 1e-10)) / X_val.shape[0]
        val_acc = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1))

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = (model.W1.copy(), model.b1.copy(), model.W2.copy(), model.b2.copy())
            patience_counter = 0
            
            # 保存最佳模型
            if save_dir:
                model.save_weights(os.path.join(save_dir, 'best_model.npz'))
        else:
            patience_counter += 1
            
        # 早停
        if early_stopping > 0 and patience_counter >= early_stopping:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, ",
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 恢复最佳权重
    if best_weights is not None:
        model.W1, model.b1, model.W2, model.b2 = best_weights

    print(f"Best Validation Accuracy: {best_acc:.4f}")

    # 可视化训练过程
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 返回训练历史
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies,
        'best_val_acc': best_acc
    }
    
    return history, best_acc


def test_neural_network(model, X_test, y_test):
    """
    在测试集上评估模型
    
    参数:
        model: 神经网络模型
        X_test, y_test: 测试数据和标签
        
    返回:
        test_acc: 测试准确率
    """
    test_preds = model.forward(X_test, is_training=False)
    test_acc = np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    绘制训练和验证的损失与准确率曲线
    
    参数:
        train_losses, val_losses: 训练和验证损失
        train_accuracies, val_accuracies: 训练和验证准确率
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')

    plt.tight_layout()
    plt.show()


def grid_search(X_train, y_train, X_val, y_val, params_grid, epochs=10, batch_size=64, 
                early_stopping=5, save_dir='checkpoints'):
    """
    对超参数进行网格搜索
    
    参数:
        X_train, y_train: 训练数据和标签
        X_val, y_val: 验证数据和标签
        params_grid: 超参数网格，格式为字典的字典
        epochs: 最大训练轮数
        batch_size: 批次大小
        early_stopping: 早停轮数
        
    返回:
        best_params: 最佳超参数组合
        best_val_acc: 最佳验证准确率
    """
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    
    # 将参数网格展开为参数组合列表
    from itertools import product
    params_keys = list(params_grid.keys())
    params_values = list(product(*[params_grid[key] for key in params_keys]))
    
    best_val_acc = 0.0
    best_params = None
    
    for i, values in enumerate(params_values):
        params = dict(zip(params_keys, values))
        print(f"\n===== 组合 {i+1}/{len(params_values)} =====")
        print(f"参数: {params}")
        
        # 初始化模型
        from mlp_modules.neural_network import NeuralNetwork
        model = NeuralNetwork(
            input_size=input_size, 
            hidden_size=params.get('hidden_size', 128),
            output_size=output_size,
            activation=params.get('activation', 'relu'),
            dropout_rate=params.get('dropout_rate', 0.5),
            use_batchnorm=params.get('use_batchnorm', True)
        )
        
        # 训练模型
        val_acc = train_neural_network(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=params.get('learning_rate', 0.01),
            lambda_reg=params.get('lambda_reg', 0.01),
            lr_decay=params.get('lr_decay', 0.9),
            momentum=params.get('momentum', 0.0),
            use_dropout=params.get('use_dropout', True),
            early_stopping=early_stopping,
            save_dir=os.path.join(save_dir, f'grid_search_{i}')
        )
        
        # 更新最佳参数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            
            # 保存最佳模型
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model.save_weights(os.path.join(save_dir, 'best_grid_search_model.npz'))
    
    print("\n===== 网格搜索完成 =====")
    print(f"最佳参数: {best_params}")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    
    return best_params, best_val_acc 