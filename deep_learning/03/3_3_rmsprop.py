import numpy

# 迭代次数
epochs = 100
# 指数加权平均
avg_squared_grad = 0
# 衰减系数
beta = 0.9
# 学习率
learning_rate = 0.01
# 目标值
target_value = 5
# 初始权重
weight = 4.4

def func(w):
    return 0.5 * (w - target_value)**2

def grad_func(w):
    return w - target_value

if __name__ == '__main__':
    epsilon = numpy.finfo(numpy.float32).eps
    for t in range(1, epochs + 1):
        grad = grad_func(weight)
        avg_squared_grad = beta * avg_squared_grad + (1 - beta) * grad**2
        weight -= learning_rate / (numpy.sqrt(avg_squared_grad) + epsilon) * grad

        # 打印结果
        if t % 10 == 0 or t == 1:
            loss = func(weight)
            print(f'Epoch {t}: weight = {weight:.4f}, loss = {loss:.4f}')
