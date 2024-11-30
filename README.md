# Machine Learning Series (机器学习系列)

📖 **深度学习 12 天速成**   [ [在线文档](https://docs.google.com/document/d/18V6H_600l-drkXd99pjNtSJtA7rIWWnER-KxIrB-lQY/edit?usp=sharing) ]  [ [YouTube 视频](https://www.youtube.com/@machine-learning-series) ]  [ 官方网站 ]  

这是一本面向初学者的深度学习综合指南。编写过程中借鉴了大量的经典教材、论文、文章，包括使用 AI 生成许多代码片段。教程主要分为四个阶段： 

**第 1 - 4 章：数学与深度学习基础**  

复习数学知识（函数、线性代数、统计学、微积分）。使用 `Keras` 高级 API 快速实现分类问题，理解什么是深度学习，包括数据、前向传播、反向传播、神经元、神经网络、优化器、损失函数、激活函数、梯度下降等基础概念。  

**第 5 - 6 章：机器学习框架入门**  

系统学习主流机器学习框架 `TensorFlow`、`PyTorch` 和 `JAX` ，加载简单数据集并完成训练，理解这些框架的相似性与差异，为后续实践打下扎实基础。  

**第 7 - 9 章：经典网络与理论提升**  

通过翻译经典论文的方式，介绍三大深度学习网络：卷积神经网络、循环神经网络、注意力机制。深度学习的发展是循序渐进的，论文可以清晰地看到人们是如何思考并解决实际问题的。  

**第 10 - 12 章：实际应用与前沿探索**  

聚焦深度学习在文本、图片和语音生成中的实际应用。探索大模型的简单实现，并详细解析 `Llama` 模型的运行与关键原理。  

文本以清晰简洁的风格编写，使其成为任何有兴趣学习深度学习的人的理想资源。


---

## [深度学习 12 天速成](https://docs.google.com/document/d/18V6H_600l-drkXd99pjNtSJtA7rIWWnER-KxIrB-lQY/edit?usp=sharing)

第 01 天 认识机器学习：绘制直线  
第 02 天 深度学习原理  
第 03 天 分类问题：Keras 求解  
第 04 天 详解反向传播算法  
第 05 天 张量与自动微分  
第 06 天 训练神经网络  
第 07 天 卷积神经网络  
第 08 天 循环神经网络  
第 09 天 注意力机制 (Transformer)  
第 10 天 生成式 (Generative)  
第 11 天 大语言模型 (LLM)  
第 12 天 前沿探索

### 01 认识机器学习：绘制直线

本章将回顾函数的基本概念，包括线性函数、多项式函数、幂函数、指数函数、对数函数、三角函数，以及函数组合、函数变换、反函数等基本性质。  

使用传统解法和机器学习解法，求一条通过 100 个随机分布点的最佳拟合直线，即找到一条直线 `y = m * x + b` 使得所有的点到直线的垂直距离之和（或平方和）最小。

![机器学习直线绘制](res/deep_learning/simplest_ml_anim.gif)

### 02 深度学习原理

了解机器学习与传统编程的区别，理解深度学习原理，熟悉人工神经网络的训练过程，包括数据预处理、权重、损失函数、优化器、反向传播等概念。

![深度学习原理](res/deep_learning/principle_deep_learning.png)

### 03 分类问题：Keras 求解

二分类问题是指在机器学习或统计学中，将数据划分为两个类别的分类任务。常见的二分类问题包括垃圾邮件分类（垃圾邮件与正常邮件）、疾病诊断（有病与无病）、图像分类（有目标与无目标）等。

```
    rng = numpy.random.default_rng(seed=0)
    input = rng.standard_normal((200, 2))
    output = numpy.array([1 if x + y > 0 else 0 for x, y in input])
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(2,)))
    model.add(keras.layers.Dense(units=1, activation='sigmoid',
                                 kernel_initializer=initializers.Constant(0.0),
                                 bias_initializer=initializers.Constant(1.0)))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
    model.fit(input, output, epochs=5, batch_size=1)
```

![二分类问题](res/deep_learning/binary_classify_anim.gif)

### 04 详解反向传播算法

复习导数（求微分）、链式法则、极值、偏导数等数学概念。通过 `NumPy` 实现常见的激活函数和损失函数，并求解它们的导数。

```
def binary_cross_entropy(y_pred, y_true):
    return -(y_true * numpy.log(y_pred) + (1 - y_true) * numpy.log(1 - y_pred))

def deriv_binary_cross_entry(y_pred, y_true):
    return y_pred - y_true
```

理解梯度和导数之间的关系，手写 **全连接神经网络 (Dense Neural Network, DNN)** ，理解网络的训练过程，即求复合函数 `h(g(f(weights, biases)))` 的极值（极大或极小）。

### 05 MNIST 全连接神经网络

MNIST 是一个入门的机器学习数据集，包含数万张手写数字 (0-9) 的灰度图像。全连接神经网络通常包含输入层、多个隐藏层和输出层，使用反向传播算法训练，从而使网络能够识别数字。

```
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = keras.models.Sequential([
        keras.layers.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit(x_train, y_train)
    model.evaluate(x_test, y_test)
```

![手写数据集样本](res/deep_learning//mnist_dataset_sample.png)

### 06 张量和自动微分

复习线性代数中的向量和矩阵。介绍主流机器学习库 (`TensorFlow/PyTorch/JAX`) 的核心内容：张量和自动微分。参考 `micrograd` 实现一个更易理解的，对标量值进行自动求导的神经网络引擎。

```
    w = tensorflow.Variable(tensorflow.fill((3, 2), 0.1), name='w')
    b = tensorflow.Variable(tensorflow.zeros(2, dtype=tensorflow.float32), name='b')
    x = [[1.0, 2.0, 3.0]]
    with tensorflow.GradientTape(persistent=True) as tape:
        y = tensorflow.math.tanh(tensorflow.matmul(x, w) + b)
        loss = tensorflow.reduce_mean(y * y)
    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
```
```
    w = torch.nn.Parameter(torch.full((3, 2), 0.1))
    b = torch.nn.Parameter(torch.zeros(2))
    x = torch.tensor([[1.0, 2.0, 3.0]])
    with torch.autograd.set_grad_enabled(True):
        y = torch.tanh(torch.matmul(x, w) + b)
        loss = torch.mean(y * y)
    loss.backward()
    dl_dw, dl_db = w.grad, b.grad
```
```
    w = jax.numpy.full((3, 2), 0.1)
    b = jax.numpy.zeros(2)
    x = jax.numpy.array([[1.0, 2.0, 3.0]])
    def forward(x, w, b):
        y = jax.numpy.tanh(jax.numpy.dot(x, w) + b)
        return jax.numpy.mean(y * y)
    grads = jax.grad(forward, argnums=(1, 2))(x, w, b)
    dl_dw, dl_db = grads
```

### 07 训练神经网络

学习常见的微型数据集。使用 `TensorFlow/PyTorch/JAX` 进行训练，理解它们的异同之处，熟练使用主流机器学习库。介绍如何对训练过程进行优化。

### 08 AlexNet 卷积模型

介绍卷积神经网络，理解卷积和池化操作。翻译著名论文 _ImageNet Classification with Deep Convolutional Neural Networks_ ，并提供代码实现。

### 09 U-Net 和 ResNet 网络

论文 _U-Net: Convolutional Networks for Biomedical Image Segmentation_ 提出大量使用数据增强的样本来训练网络的策略，相比于传统的卷积网络，需要更少的数据集，但是效果却更好。  
论文 _Deep Residual Learning for Image Recognition_ 提出一个残差学习框架 (Residual Network) ，可以很容易训练比以前更深的网络，具有更高的准确性。

### 10 注意力机制 (Transformer)

介绍循环神经网络，认识它和前馈网络的不同之处。翻译著名论文 _Attention Is All You Need_ ，并作出详细的解释，彻底理解 `Transformer` 架构。

### 11 生成式 (Generative)

生成式方法的目标是在已知的样本数据上学习其特征分布，然后生成具有相似特征的全新数据，包括：稳定扩撒、神经风格迁移、DeepDream、卷积生成对抗网络、Pix2Pix 、CycleGAN 。

### 12 大语言模型 (LLM)

`nanoGPT` 是最简单、最快的中型 GPT 训练/微调存储库，优先考虑实用性而非教育性。介绍 Llama 开源模型，包括如何访问模型、托管、操作方法和集成指南。

---

* Python 语言 12 天速成 [ [在线文档](https://docs.google.com/document/d/13dJIhnj4FbxFApRbaxyYz436vsRMAK9FhqPyuqBMY9Q/edit?usp=sharing) ] [ [YouTube 视频](https://www.youtube.com/@machine-learning-series) ]

## [Python 语言 12 天速成](https://docs.google.com/document/d/13dJIhnj4FbxFApRbaxyYz436vsRMAK9FhqPyuqBMY9Q/edit?usp=sharing)

第 1 天 Python 介绍  
第 2 天 计算机基础  
第 3 天 常量与变量  
第 4 天 控制流  
第 5 天 函数详解  
第 6 天 实战：Android 控制  
第 7 天 数据结构与算法  
第 8 天 面向对象  
第 9 天 标准库  
第 10 天 实战：绘制跳动的爱心  
第 11 天 实战：WAVE 音频解析  
第 12 天 机器学习常用库  

---
