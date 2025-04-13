import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mlp_modules.data_utils import load_CIFAR10_data, split_train_val
from mlp_modules.neural_network import NeuralNetwork
from mlp_modules.train_utils import (
    train_neural_network, 
    train_neural_network_with_metrics,
    test_neural_network,
    grid_search
)

def main():
    """主函数，处理命令行参数并运行相应的功能"""
    parser = argparse.ArgumentParser(description="训练CIFAR-10分类器的神经网络")
    
    # 基本参数
    parser.add_argument('--data_dir', type=str, required=True, help='CIFAR-10数据集路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'grid_search'], 
                       help='运行模式：训练、测试或网格搜索')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=1024, help='隐藏层大小')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'], 
                       help='激活函数类型')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout比率')
    parser.add_argument('--use_batchnorm', action='store_true', help='是否使用批归一化')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='学习率衰减率')
    parser.add_argument('--lambda_reg', type=float, default=0.01, help='L2正则化系数')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量系数')
    parser.add_argument('--use_dropout', action='store_true', help='是否使用dropout')
    parser.add_argument('--early_stopping', type=int, default=10, help='早停轮数')
    parser.add_argument('--model_path', type=str, default=None, help='预训练模型路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 加载数据
    print("加载 CIFAR-10 数据...")
    X_train_full, y_train_full, X_test, y_test = load_CIFAR10_data(args.data_dir)
    
    # 划分训练集和验证集
    print("划分训练集和验证集...")
    X_train, y_train, X_val, y_val = split_train_val(X_train_full, y_train_full, val_ratio=0.2)
    
    # 打印数据集信息
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        # 初始化模型
        model = NeuralNetwork(
            input_size=X_train.shape[1], 
            hidden_size=args.hidden_size,
            output_size=y_train.shape[1],
            activation=args.activation,
            dropout_rate=args.dropout_rate,
            use_batchnorm=args.use_batchnorm
        )
        
        # 如果有预训练模型，加载它
        if args.model_path and os.path.exists(args.model_path):
            print(f"加载预训练模型: {args.model_path}")
            model.load_weights(args.model_path)
            
        # 训练模型
        print("开始训练...")
        history, best_val_acc = train_neural_network_with_metrics(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lambda_reg=args.lambda_reg,
            lr_decay=args.lr_decay,
            momentum=args.momentum,
            use_dropout=args.use_dropout,
            early_stopping=args.early_stopping,
            save_dir=args.output_dir
        )
        
        # 测试模型
        print("在测试集上评估...")
        test_acc = test_neural_network(model, X_test, y_test)
        
    elif args.mode == 'test':
        # 检查模型路径
        if not args.model_path or not os.path.exists(args.model_path):
            raise ValueError("测试模式需要提供有效的模型路径")
            
        # 初始化模型
        model = NeuralNetwork(
            input_size=X_train.shape[1], 
            hidden_size=args.hidden_size,
            output_size=y_train.shape[1],
            activation=args.activation,
            dropout_rate=args.dropout_rate,
            use_batchnorm=args.use_batchnorm
        )
        
        # 加载模型
        print(f"加载模型: {args.model_path}")
        model.load_weights(args.model_path)
        
        # 测试模型
        print("在测试集上评估...")
        test_acc = test_neural_network(model, X_test, y_test)
        
    elif args.mode == 'grid_search':
        # 定义超参数网格
        params_grid = {
            'hidden_size': [512, 1024, 2048],
            'learning_rate': [0.01, 0.001],
            'lambda_reg': [0.0, 0.01, 0.001],
            'dropout_rate': [0.3, 0.5],
            'momentum': [0.0, 0.9],
            'use_batchnorm': [True, False]
        }
        
        # 执行网格搜索
        print("开始网格搜索...")
        best_params, best_val_acc = grid_search(
            X_train, y_train, X_val, y_val,
            params_grid=params_grid,
            epochs=args.epochs,
            batch_size=args.batch_size,
            early_stopping=args.early_stopping,
            save_dir=args.output_dir
        )
        
        # 使用最佳参数初始化模型
        print("使用最佳参数训练最终模型...")
        model = NeuralNetwork(
            input_size=X_train.shape[1], 
            hidden_size=best_params['hidden_size'],
            output_size=y_train.shape[1],
            activation=best_params['activation'],
            dropout_rate=best_params['dropout_rate'],
            use_batchnorm=best_params['use_batchnorm']
        )
        
        # 加载网格搜索找到的最佳模型
        model.load_weights(os.path.join(args.output_dir, 'best_grid_search_model.npz'))
        
        # 测试模型
        print("在测试集上评估...")
        test_acc = test_neural_network(model, X_test, y_test)


if __name__ == "__main__":
    main() 