import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', dropout_rate=0.5, use_batchnorm=True):
        """
        初始化神经网络结构: input->hidden->output
        
        参数:
            input_size: 输入层大小
            hidden_size: 隐藏层大小
            output_size: 输出层大小
            activation: 激活函数类型，默认为'relu'
            dropout_rate: dropout比率，默认为0.5
            use_batchnorm: 是否使用批归一化，默认为True
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm

        # 参数初始化（He初始化）
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        # 批归一化参数
        if self.use_batchnorm:
            # gamma（缩放因子）和beta（平移因子）
            self.gamma1 = np.ones((1, hidden_size))
            self.beta1 = np.zeros((1, hidden_size))
            
            # 用于测试时的滑动平均
            self.running_mean1 = np.zeros((1, hidden_size))
            self.running_var1 = np.ones((1, hidden_size))
            self.momentum = 0.9  # 滑动平均动量因子

        # 动量参数
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        
        if self.use_batchnorm:
            self.vgamma1 = np.zeros_like(self.gamma1)
            self.vbeta1 = np.zeros_like(self.beta1)

        # 存储中间结果
        self.cache = {}

    def relu(self, z):
        """ReLU激活函数"""
        return np.maximum(0, z)

    def softmax(self, z):
        """Softmax激活函数（数值稳定版本）"""
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def batch_norm_forward(self, x, gamma, beta, bn_param, eps=1e-5):
        """
        批归一化前向传播
        
        参数:
            x: 输入数据
            gamma: 缩放参数
            beta: 平移参数
            bn_param: 批归一化参数字典
            eps: 数值稳定性常数
            
        返回:
            out: 归一化后的输出
            cache: 缓存中间值用于反向传播
        """
        mode = bn_param.get('mode', 'train')
        momentum = bn_param.get('momentum', 0.9)
        running_mean = bn_param.get('running_mean', np.zeros_like(x[0]))
        running_var = bn_param.get('running_var', np.ones_like(x[0]))
        
        if mode == 'train':
            # 计算均值和方差
            sample_mean = np.mean(x, axis=0, keepdims=True)
            sample_var = np.var(x, axis=0, keepdims=True)
            
            # 归一化
            x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)
            
            # 缩放和平移
            out = gamma * x_norm + beta
            
            # 更新滑动平均
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var
            
            # 缓存用于反向传播
            cache = {
                'x': x,
                'x_norm': x_norm,
                'mean': sample_mean,
                'var': sample_var,
                'gamma': gamma,
                'beta': beta,
                'eps': eps
            }
        elif mode == 'test':
            # 测试时使用滑动平均
            x_norm = (x - running_mean) / np.sqrt(running_var + eps)
            out = gamma * x_norm + beta
            cache = None
        else:
            raise ValueError(f'Invalid batch normalization mode: {mode}')
        
        # 更新运行时参数
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        
        return out, cache, bn_param
    
    def batch_norm_backward(self, dout, cache):
        """
        批归一化反向传播
        
        参数:
            dout: 上游梯度
            cache: 前向传播缓存
            
        返回:
            dx: 输入梯度
            dgamma: gamma梯度
            dbeta: beta梯度
        """
        x = cache['x']
        x_norm = cache['x_norm']
        mean = cache['mean']
        var = cache['var']
        gamma = cache['gamma']
        eps = cache['eps']
        
        N = x.shape[0]
        
        # beta梯度就是dout的和
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # gamma梯度
        dgamma = np.sum(dout * x_norm, axis=0, keepdims=True)
        
        # 计算dx_norm
        dx_norm = dout * gamma
        
        # 计算dvar
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * np.power(var + eps, -1.5), axis=0, keepdims=True)
        
        # 计算dmean
        dmean = np.sum(dx_norm * -1.0 / np.sqrt(var + eps), axis=0, keepdims=True)
        dmean += dvar * np.sum(-2.0 * (x - mean), axis=0, keepdims=True) / N
        
        # 计算dx
        dx = dx_norm / np.sqrt(var + eps)
        dx += dvar * 2.0 * (x - mean) / N
        dx += dmean / N
        
        return dx, dgamma, dbeta
    
    def dropout(self, X, rate, is_training=True):
        """
        实现Dropout功能
        
        参数:
            X: 输入张量
            rate: dropout概率（舍弃的比例）
            is_training: 是否处于训练模式
            
        返回:
            带有dropout效果的输出
        """
        if not is_training:
            return X
        
        # 生成dropout掩码
        mask = np.random.binomial(1, 1-rate, size=X.shape) / (1-rate)
        # 应用掩码
        return X * mask

    def forward(self, X, is_training=False):
        """
        前向传播
        
        参数:
            X: 输入数据
            is_training: 是否处于训练模式（影响dropout和批归一化行为）
            
        返回:
            输出层激活值
        """
        # 创建批归一化参数
        bn_param1 = {}
        bn_param1['mode'] = 'train' if is_training else 'test'
        bn_param1['momentum'] = self.momentum
        bn_param1['running_mean'] = self.running_mean1 if self.use_batchnorm else None
        bn_param1['running_var'] = self.running_var1 if self.use_batchnorm else None
        
        # 第一层线性变换
        z1 = np.dot(X, self.W1) + self.b1
        
        # 批归一化（如果启用）
        if self.use_batchnorm:
            z1_bn, bn_cache1, bn_param1_updated = self.batch_norm_forward(
                z1, self.gamma1, self.beta1, bn_param1
            )
            # 更新运行时统计量
            if is_training:
                self.running_mean1 = bn_param1_updated['running_mean']
                self.running_var1 = bn_param1_updated['running_var']
            a1 = self.relu(z1_bn) if self.activation == 'relu' else z1_bn
        else:
            a1 = self.relu(z1) if self.activation == 'relu' else z1
            bn_cache1 = None
        
        # Dropout - 只在训练时应用
        if is_training:
            a1 = self.dropout(a1, self.dropout_rate, is_training)
        
        # 第二层
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)

        # 缓存
        self.cache['X'] = X
        self.cache['z1'] = z1
        self.cache['z1_bn'] = z1_bn if self.use_batchnorm else None
        self.cache['bn_cache1'] = bn_cache1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        self.cache['dropout_mask'] = a1 / (self.relu(z1_bn if self.use_batchnorm else z1) 
                                         if self.activation == 'relu' else (z1_bn if self.use_batchnorm else z1)) if is_training else None

        return a2

    def backward(self, X, y, learning_rate=0.01, lambda_reg=0.01, momentum=0.0):
        """
        反向传播并更新参数
        
        参数:
            X: 输入数据
            y: 目标标签
            learning_rate: 学习率
            lambda_reg: L2正则化系数
            momentum: 动量系数
            
        返回:
            无返回值，但会更新网络参数
        """
        m = X.shape[0]
        a1 = self.cache['a1']
        a2 = self.cache['a2']
        z1 = self.cache['z1']
        bn_cache1 = self.cache['bn_cache1']

        # 输出层梯度
        dz2 = a2 - y  # (N, output_size)
        dW2 = np.dot(a1.T, dz2) / m + lambda_reg * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # 隐藏层梯度
        dz1_next = np.dot(dz2, self.W2.T)
        
        # 如果使用了dropout，应用相同的掩码
        if 'dropout_mask' in self.cache and self.cache['dropout_mask'] is not None:
            dz1_next *= self.cache['dropout_mask']
            
        # 批归一化的反向传播
        if self.use_batchnorm:
            dz1_bn = dz1_next.copy()
            # ReLU的导数：dReLU = 1(z>0), 0(z<=0)
            dz1_bn[self.cache['z1_bn'] <= 0] = 0  # 针对ReLU
            
            # 批归一化梯度
            dz1, dgamma1, dbeta1 = self.batch_norm_backward(dz1_bn, bn_cache1)
        else:
            dz1 = dz1_next.copy()
            # ReLU的导数：dReLU = 1(z>0), 0(z<=0)
            dz1[z1 <= 0] = 0  # 针对ReLU
            
        dW1 = np.dot(X.T, dz1) / m + lambda_reg * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 使用动量更新参数
        if momentum > 0:
            self.vW1 = momentum * self.vW1 - learning_rate * dW1
            self.vb1 = momentum * self.vb1 - learning_rate * db1
            self.vW2 = momentum * self.vW2 - learning_rate * dW2
            self.vb2 = momentum * self.vb2 - learning_rate * db2
            
            if self.use_batchnorm:
                self.vgamma1 = momentum * self.vgamma1 - learning_rate * dgamma1
                self.vbeta1 = momentum * self.vbeta1 - learning_rate * dbeta1
            
            self.W1 += self.vW1
            self.b1 += self.vb1
            self.W2 += self.vW2
            self.b2 += self.vb2
            
            if self.use_batchnorm:
                self.gamma1 += self.vgamma1
                self.beta1 += self.vbeta1
        else:
            # 不使用动量的普通SGD更新
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            
            if self.use_batchnorm:
                self.gamma1 -= learning_rate * dgamma1
                self.beta1 -= learning_rate * dbeta1
    
    def save_weights(self, filepath):
        """保存模型权重到文件"""
        if self.use_batchnorm:
            np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                    gamma1=self.gamma1, beta1=self.beta1,
                    running_mean1=self.running_mean1, running_var1=self.running_var1)
        else:
            np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
    
    def load_weights(self, filepath):
        """从文件加载模型权重"""
        weights = np.load(filepath)
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']
        
        if self.use_batchnorm and 'gamma1' in weights:
            self.gamma1 = weights['gamma1']
            self.beta1 = weights['beta1']
            self.running_mean1 = weights['running_mean1']
            self.running_var1 = weights['running_var1'] 