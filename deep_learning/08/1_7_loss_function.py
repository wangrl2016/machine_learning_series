import numpy

if __name__ == '__main__':
    pred = [0.1, 0.9]
    # 假设真实分布为 [0, 1]
    # 正向情绪
    loss = - numpy.log(0.9)
    print(loss)
    # 假设真实分布为 [1, 0]
    # 负向情绪
    loss = - numpy.log(0.1)
    print(loss)
