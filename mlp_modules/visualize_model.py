#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

# 添加中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from mlp_modules.visualize_utils import visualize_weights


def main():
    """主函数，解析命令行参数并运行模型可视化"""
    parser = argparse.ArgumentParser(description="可视化神经网络模型参数")
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型权重文件路径(.npz格式)')
    
    # 可选参数
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                        help='可视化结果保存目录')
    parser.add_argument('--save_results', action='store_true',
                        help='是否保存可视化结果')
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"找不到模型文件: {args.model_path}")
    
    # 创建输出目录（如果需要保存结果）
    output_dir = args.output_dir if args.save_results else None
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # CIFAR-10类别名称（中文）
    cifar10_classes = [
        '飞机', '汽车', '鸟', '猫', '鹿',
        '狗', '青蛙', '马', '船', '卡车'
    ]
    
    # 开始可视化
    print(f"开始可视化模型: {args.model_path}")
    visualize_weights(args.model_path, output_dir, cifar10_classes)
    
    if output_dir:
        print(f"可视化结果已保存到: {output_dir}")


if __name__ == "__main__":
    main() 