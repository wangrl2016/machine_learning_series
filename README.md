# Machine Learning Series (机器学习系列)

* 机器学习 12 天速成  [ [在线文档](https://docs.google.com/document/d/18V6H_600l-drkXd99pjNtSJtA7rIWWnER-KxIrB-lQY/edit?usp=sharing) ] [ [YouTube 视频](https://www.youtube.com/@machine-learning-series) ]  

* Python 语言 12 天速成 [ [在线文档](https://docs.google.com/document/d/13dJIhnj4FbxFApRbaxyYz436vsRMAK9FhqPyuqBMY9Q/edit?usp=sharing) ] [ [YouTube 视频](https://www.youtube.com/@machine-learning-series) ]

---

## [机器学习 12 天速成](https://docs.google.com/document/d/18V6H_600l-drkXd99pjNtSJtA7rIWWnER-KxIrB-lQY/edit?usp=sharing)

第 1 天 认识机器学习：绘制直线  
第 2 天 深度学习原理  
第 3 天 二分类问题：Keras 求解  
第 4 天 详解反向传播算法  
第 5 天 MNIST 全连接神经网络  
第 6 天 张量与自动微分  
第 7 天 训练神经网络  
第 8 天 AlexNet 卷积模型  
第 9 天 U-Net 和 ResNet 网络  
第 10 天 注意力机制 (Transformer)  
第 11 天 生成式 (Generative)  
第 12 天 大语言模型 (LLM)  

### 01 认识机器学习：绘制直线
本章将回顾常用函数的基本概念，使用传统解法和机器学习解法，求一条通过 100 个随机分布点的最佳拟合直线，即找到一条直线 `y = m * x + b` 使得所有的点到直线的垂直距离之和（或平方和）最小。

![机器学习直线绘制](res/machine_learning/simplest_ml_anim.gif)

### 02 深度学习原理

了解机器学习与传统编程的区别，理解深度学习原理，熟悉人工神经网络的训练过程，包括数据预处理、权重、损失函数、优化器、反向传播等概念。

![深度学习原理](res/machine_learning/principle_deep_learning.png)

### 03 二分类问题：Keras 求解

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

![二分类问题](res/machine_learning/binary_classify_anim.gif)

### 04 详解反向传播算法

复习导数（求微分）、链式法则、极值、偏导数等数学概念。理解梯度和导数之间的关系，手写人工神经网络，求解函数 `loss = h(g(f(weights, biases)))` 的最小值（训练网络）。

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

![手写数据集样本](res/machine_learning//mnist_dataset_sample.png)

### 06 张量和自动微分

复习线性代数中的向量和矩阵。介绍主流机器学习库 (`TensorFlow/PyTorch/JAX`) 的核心内容：张量和自动微分。实现一个对标量值进行自动求导的神经网络引擎。

### 07 训练神经网络

学习常见的微型数据集。使用 `TensorFlow/PyTorch/JAX` 进行训练，理解它们的异同之处，熟练使用主流机器学习库。介绍如何对训练过程进行优化。

### 08 AlexNet 卷积模型

介绍卷积神经网络，认识卷积和池化操作。翻译著名论文 _ImageNet Classification with Deep Convolutional Neural Networks_ ，并提供代码实现。

### 09 U-Net 和 ResNet 网络

论文 _U-Net: Convolutional Networks for Biomedical Image Segmentation_ 提出大量使用数据增强的样本来训练网络的策略，相比于传统的卷积网络，需要更少的数据集，但是效果却更好。  
论文 _Deep Residual Learning for Image Recognition_ 提出一个残差学习框架 (Residual Network) ，可以很容易训练比以前更深的网络，具有更高的准确性。

### 10 注意力机制 (Transformer)

介绍循环神经网络，认识它和前馈网络的不同之处。翻译著名论文 _Attention Is All You Need_ ，并作出详细的解释。

### 11 生成式 (Generative)

生成式方法的目标是在已知的样本数据上学习其特征分布，然后生成具有相似特征的全新数据，包括：稳定扩撒、神经风格迁移、DeepDream、卷积生成对抗网络、Pix2Pix 、CycleGAN 。

### 12 大语言模型 (LLM)

nanoGPT 是最简单、最快的中型 GPT 训练/微调存储库，优先考虑实用性而非教育性。介绍 Llama 开源模型，包括如何访问模型、托管、操作方法和集成指南。

---

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
