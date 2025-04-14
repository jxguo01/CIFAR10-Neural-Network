# CIFAR-10 三层神经网络分类器

基于NumPy实现的三层神经网络，用于CIFAR-10图像分类任务。该项目通过手动实现前向传播和反向传播算法，实现了神经网络的完整训练和测试流程。

## 项目特点

- 纯NumPy实现的三层神经网络，无需深度学习框架
- 支持多种优化和正则化技术：
  - Dropout防止过拟合
  - 批归一化(Batch Normalization)
  - L2正则化
  - 基于动量的SGD优化器
  - 多种学习率衰减策略
  - 早停机制
- 模块化设计，便于扩展和修改
- 提供完整的训练、验证和测试流程
- 支持超参数网格搜索
- 自动保存最佳模型权重

## 项目结构

```
├── mlp_modules/                  # 主要代码模块
│   ├── __init__.py              # 包初始化文件
│   ├── data_utils.py            # 数据加载和预处理
│   ├── neural_network.py        # 神经网络模型实现
│   ├── train_utils.py           # 训练和评估功能
│   ├── main.py                  # 命令行接口和主程序
│   ├── visualize_model.py       # 模型可视化命令行工具
│   └── visualize_utils.py       # 模型可视化核心功能
├── run.py                        # 快速启动脚本
└── README.md                     # 项目说明文件
```

## 主要功能

1. **数据处理** (`data_utils.py`)
   - CIFAR-10数据集加载和预处理
   - 训练集/验证集分割
   - 数据增强（随机水平翻转）

2. **神经网络模型** (`neural_network.py`)
   - 输入层 -> 隐藏层 -> 输出层结构
   - 支持ReLU激活函数
   - Dropout正则化
   - 手动实现前向和反向传播

3. **训练和评估** (`train_utils.py`)
   - 小批量随机梯度下降(SGD)
   - 动量优化
   - 学习率衰减（自定义衰减策略）
   - 损失和准确率可视化
   - 超参数网格搜索

4. **命令行工具** (`main.py`)
   - 训练模式
   - 测试模式
   - 超参数搜索模式

## 安装要求

```bash
pip install numpy matplotlib scikit-learn
```

## 数据集准备

本项目使用CIFAR-10数据集，您需要首先下载并解压该数据集：

1. 从[官方网站](https://www.cs.toronto.edu/~kriz/cifar.html)下载CIFAR-10数据集的Python版本
2. 解压下载的文件，得到`cifar-10-batches-py`文件夹
3. 记下该文件夹的完整路径，将在训练和测试命令中使用

文件夹结构应该如下：
```
cifar-10-batches-py/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```

## 训练和测试指南

### 训练模型

训练神经网络模型的基本步骤如下：

1. 确保已准备好CIFAR-10数据集
2. 执行训练命令，指定必要参数

基本训练命令：
```bash
python run.py --data_dir /path/to/cifar-10-batches-py --mode train
```

使用推荐的参数进行训练：
```bash
python run.py --data_dir /path/to/cifar-10-batches-py --mode train --hidden_size 1024 --epochs 100 --batch_size 128 --learning_rate 0.01 --lambda_reg 0.01 --momentum 0.9 --use_dropout
```

使用批归一化进行训练：
```bash
python run.py --data_dir /path/to/cifar-10-batches-py --mode train --hidden_size 1024 --epochs 100 --batch_size 128 --learning_rate 0.01 --lambda_reg 0.01 --momentum 0.9 --use_dropout --use_batchnorm
```

训练过程将显示每个epoch的训练损失、训练准确率、验证损失和验证准确率。同时，会自动保存验证集准确率最高的模型到`outputs/best_model.npz`。

训练结束后，您将看到类似以下输出：
```
Epoch 100/100 | Train Loss: 1.1234, Train Acc: 0.6789, Val Loss: 1.2345, Val Acc: 0.5678
Best Validation Accuracy: 0.5832
```

同时，会显示训练过程中的损失和准确率曲线图。

### 测试模型

在测试模型之前，确保您已经：
1. 完成了模型训练，或有一个预训练的模型文件
2. 准备好CIFAR-10测试数据

执行测试命令：
```bash
python run.py --data_dir /path/to/cifar-10-batches-py --mode test --model_path outputs/best_model.npz --hidden_size 1024
```

### 超参数搜索

```bash
python run.py --data_dir /path/to/cifar-10-batches-py --mode grid_search --epochs 20
```

## 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_dir` | CIFAR-10数据集路径 | (必需) |
| `--output_dir` | 输出目录 | outputs |
| `--mode` | 运行模式 [train/test/grid_search] | train |
| `--hidden_size` | 隐藏层大小 | 1024 |
| `--activation` | 激活函数 [relu] | relu |
| `--dropout_rate` | Dropout比率 | 0.5 |
| `--use_batchnorm` | 是否使用批归一化 | false |
| `--epochs` | 训练轮数 | 100 |
| `--batch_size` | 批次大小 | 128 |
| `--learning_rate` | 初始学习率 | 0.01 |
| `--lr_decay` | 学习率衰减率 | 0.9 |
| `--lambda_reg` | L2正则化系数 | 0.01 |
| `--momentum` | 动量系数 | 0.9 |
| `--use_dropout` | 是否使用dropout | false |
| `--early_stopping` | 早停轮数 | 10 |
| `--model_path` | 预训练模型路径 | None |

## 学习率调整策略

本项目实现了两种学习率调整策略的结合：

1. **阶梯式衰减**：当轮数 ≥ 5且轮数是10的倍数时，学习率乘以0.5
2. **指数衰减**：每轮结束后，学习率乘以衰减率(lr_decay)

## 模型保存

- 训练过程中，基于验证集准确率自动保存最佳模型
- 模型保存路径为`{output_dir}/best_model.npz`
- 保存的模型包含权重矩阵(W1, W2)和偏置向量(b1, b2)

## 实验结果

通过超参数调优，在CIFAR-10数据集上可以获得约58.32%及以上的测试准确率 

## 模型可视化工具

本项目提供了一套神经网络模型参数可视化工具，可帮助理解模型学习到的特征和权重分布。

### 功能特点

- 隐藏层神经元滤波器可视化
- 输入到隐藏层权重(W1)分布分析
- 隐藏层到输出层权重(W2)热力图
- 批归一化参数可视化
- 权重统计信息(均值、方差、分布)
- 各类别重要特征可视化

### 使用方法

```bash
python -m mlp_modules.visualize_model --model_path outputs/best_model.npz --save_results
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型权重文件路径(.npz格式) | (必需) |
| `--output_dir` | 可视化结果保存目录 | visualization_results |
| `--save_results` | 是否保存可视化结果 | false |

### 中文显示支持

可视化工具已配置支持中文字体显示，使用了以下字体配置:
- SimHei (黑体)
- DejaVu Sans
- Arial Unicode MS
- Microsoft YaHei (微软雅黑)

如果图表中的中文仍无法正常显示，请确保系统已安装上述字体之一。 
