import tensorflow

LAYER_SIZES = [784, 256, 128, 10]

if __name__ == '__main__':
    print('TensorFlow version:', tensorflow.__version__)
    weights = {
        'h1': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[0], LAYER_SIZES[1]])),
        'h2': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[1], LAYER_SIZES[2]])),
        'out': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[2], LAYER_SIZES[3]])),}
    biases = {
        'b1': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[1]])),
        'b2': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[2]])),
        'out': tensorflow.Variable(tensorflow.random.normal([LAYER_SIZES[3]])),}
    
    # 定义网络结构
    def NeuralNetwork(x):
        layer_1 = tensorflow.add(tensorflow.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tensorflow.nn.relu(layer_1)
        layer_2 = tensorflow.add(tensorflow.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tensorflow.nn.relu(layer_2)
        out_layer = tensorflow.matmul(layer_2, weights['out']) + biases['out']
    