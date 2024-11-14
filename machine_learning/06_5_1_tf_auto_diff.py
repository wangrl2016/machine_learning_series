import tensorflow

if __name__ == '__main__':
    x = tensorflow.Variable(3.0)
    with tensorflow.GradientTape() as tape:
        y = x**2
    
    # dy = 2x * dx
    dy_dx = tape.gradient(y, x)
    print(dy_dx.numpy())

    w = tensorflow.Variable(tensorflow.fill((3, 2), 0.1), name='w')
    b = tensorflow.Variable(tensorflow.zeros(2, dtype=tensorflow.float32), name='b')
    x = [[1.0, 2.0, 3.0]]
    with tensorflow.GradientTape(persistent=True) as tape:
        y = tensorflow.math.tanh(tensorflow.matmul(x, w) + b)
        loss = tensorflow.reduce_mean(y * y)
    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
    print(dl_dw)
    print(dl_db)

