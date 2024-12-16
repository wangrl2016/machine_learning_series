# Machine Learning Series (机器学习系列)

* Python 语言 12 天速成

* 深度学习 12 天速成

---

## 📖 [深度学习 12 天速成](https://docs.google.com/document/d/18V6H_600l-drkXd99pjNtSJtA7rIWWnER-KxIrB-lQY/edit?usp=sharing)

[ [在线文档](https://docs.google.com/document/d/18V6H_600l-drkXd99pjNtSJtA7rIWWnER-KxIrB-lQY/edit?usp=sharing) ]  [ [YouTube 视频](https://www.youtube.com/@machine-learning-series) ]  [ 官方网站 ]  

这是一本面向初学者的深度学习综合指南。编写过程中借鉴了大量的经典教材、论文、文章，包括使用 AI 生成许多代码片段。教程主要分为四个阶段： 

**第 1 - 4 章：数学与深度学习基础**  

复习数学知识（函数、线性代数、统计学、微积分）。使用 `Keras` 高级 API 快速实现分类问题，理解什么是深度学习，包括数据、前向传播、反向传播、神经元、神经网络、优化器、损失函数、激活函数、梯度下降等基础概念。  

**第 5 - 6 章：机器学习框架入门**  

系统学习主流机器学习框架 `TensorFlow`、`PyTorch` 和 `JAX` ，熟练使用张量和自动微分，加载简单数据集并完成训练，理解这些框架的相似性与差异，为后续实践打下扎实基础。  

**第 7 - 9 章：经典网络与理论提升**  

通过翻译经典论文的方式，介绍三大深度学习网络：卷积神经网络、循环神经网络、注意力机制。深度学习的发展是循序渐进的，论文可以清晰地看到人们是如何思考，并解决实际问题的。  

**第 10 - 12 章：实际应用与前沿探索**  

聚焦深度学习在文本、图片和语音生成中的实际应用，介绍稳定扩散、生成对抗网络等技术。探索大模型的简单实现，并详细解析 `Llama` 模型的运行与关键原理。  

文本以清晰简洁的风格编写，使其成为任何有兴趣学习深度学习的人的理想资源。

---

### 01 认识机器学习：绘制直线

本章将回顾函数的基本概念，包括线性函数、幂函数、多项式函数、有理函数、指数函数、对数函数、三角函数，以及函数组合、函数变换、反函数等基本性质。  

![二次曲线变换动画](res/deep_learning/shift_anim.gif)

详细介绍 `NumPy` 科学计算库，使用各种方法创建 `ndarray` 数组，对数组进行索引和切片，并探讨数组之间的计算，例如广播、连接、乘法等运算。  

使用传统解法和机器学习解法，求一条通过 100 个随机分布点的最佳拟合直线，即找到一条直线 `y = m * x + b` 使得所有的点到直线的垂直距离之和（或平方和）最小。  

![机器学习直线绘制](res/deep_learning/simplest_ml_anim.gif)

### 02 深度学习原理

了解机器学习与传统编程的区别，理解深度学习原理，熟悉人工神经网络的训练过程，包括数据预处理、神经元、权重、损失函数、优化器、反向传播等概念。  

![深度学习原理](res/deep_learning/principle_deep_learning.png)

教材《线性代数介绍》深入浅出地介绍了线性代数的核心概念，包括矩阵运算、向量空间、线性变换、正交性、特征值与特征向量等。  

将日常生活中的常见表示（特征数据、文字、图片、视频、声音）转换为神经网络的数据输入，对数据进行一些预处理操作，手动实现常见的数据处理算法。  

![手写数据集样本](res/deep_learning//mnist_dataset_sample.png)

### 03 分类问题：Keras 求解

介绍统计学的基础知识，包括采样、数据统计、概率、正态分布等。使用标准正态函数，生成分类问题的随机分布点。  

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

多分类问题是机器学习中的一个常见任务，其目标是将输入数据分配到多个类别中的一个。例如给定一张图片，模型需要判断图片中的内容是猫、狗还是鸟。  

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

本章主要使用 Keras 高级 API 来解决以上两个问题，进一步熟悉深度学习中常见的模块，包括模型、层、损失函数、优化器等。通过快速上手简单的示例，理解深度学习全流程，为后续详细介绍奠定基础。  

![随机梯度优化器](res/deep_learning/sgd_momentum.gif)

### 04 详解反向传播算法

复习导数（求微分）、链式法则、极值、偏导数等数学概念。通过 `NumPy` 实现常见的激活函数和损失函数，并求解它们的导数。使用链式法则求解模型的梯度，理解权重是如何更新的。  

```
def binary_cross_entropy(y_pred, y_true):
    return -(y_true * numpy.log(y_pred) + (1 - y_true) * numpy.log(1 - y_pred))

def deriv_binary_cross_entry(y_pred, y_true):
    return y_pred - y_true
```

手写**全连接神经网络 (Dense Neural Network, DNN)** ，理解网络的训练过程，即求复合函数 `h(g(f(weights, biases)))` 的极值（极大或极小），实现几个简单的模型。  

### 05 张量和自动微分

介绍主流机器学习库 (TensorFlow/PyTorch/JAX) 的核心内容：张量和自动微分。张量是机器学习中的一种核心数据结构，它可以看作是多维数组。自动微分是机器学习库用来自动计算梯度的工具。  

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

实现一个对标量值进行自动求导的神经网络引擎 micrograd ，它主要用于构建简单的神经网络并计算梯度，使用它构建深度神经网络，进行二分类展示。  

分析 tinygrad 源码，不仅是教育目的（理解深度学习框架的底层原理），相比 micrograd 还增加了多维张量支持和更丰富的功能，使其接近实际深度学习框架的核心工作方式。  

### 06 训练神经网络



### 07 卷积神经网络

介绍卷积神经网络，理解卷积和池化操作。卷积操作通过滑动卷积核对输入进行加权求和，提取局部特征，如边缘或纹理。池化操作则通过选择局部区域的最大值或平均值，减少图像的尺寸。  

论文 _ImageNet Classification with Deep Convolutional Neural Networks_ 使用 ReLU 激活函数，利用 GPU 加速训练，证明深度学习在大规模图像数据上的潜力，成为现代深度学习崛起的里程碑。  

论文 _U-Net: Convolutional Networks for Biomedical Image Segmentation_ 采用对称的编码-解码结构，提出了有效的小样本训练方法。广泛应用于医学图像分析领域，例如肿瘤检测、器官分割等任务。  

论文 _Deep Residual Learning for Image Recognition_ 使用残差模块，使网络深度达到数百甚至上千层，同时提升性能。成为深度网络的基础架构，被广泛应用于图像分类、目标检测、语义分割等任务。  

通过翻译论文的形式，可以深入理解卷积神经网络的发展历程及其在各个领域的应用，为后续的深入学习打下基础。  

### 08 循环神经网络

### 09 注意力机制 (Transformer)

认识它和前馈网络的不同之处。翻译著名论文 _Attention Is All You Need_ ，并作出详细的解释，彻底理解 `Transformer` 架构。

### 10 生成式 (Generative)

生成式方法的目标是在已知的样本数据上学习其特征分布，然后生成具有相似特征的全新数据，包括：稳定扩撒、神经风格迁移、DeepDream、卷积生成对抗网络、Pix2Pix 、CycleGAN 。

### 11 大语言模型 (LLM)

`nanoGPT` 是最简单、最快的中型 GPT 训练/微调存储库，优先考虑实用性而非教育性。介绍 Llama 开源模型，包括如何访问模型、托管、操作方法和集成指南。

### 12 前沿探索

---

## 📖 [Python 语言 12 天速成](https://docs.google.com/document/d/13dJIhnj4FbxFApRbaxyYz436vsRMAK9FhqPyuqBMY9Q/edit?usp=sharing)

[ [在线文档](https://docs.google.com/document/d/13dJIhnj4FbxFApRbaxyYz436vsRMAK9FhqPyuqBMY9Q/edit?usp=sharing) ] [ [YouTube 视频](https://www.youtube.com/@machine-learning-series) ]

---
