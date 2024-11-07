import numpy
from matplotlib import pyplot

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    def __init__(self):
        rng = numpy.random.default_rng(0)
        # weights
        self.w1 = rng.random()
        self.w2 = rng.random()
        self.w3 = rng.random()
        self.w4 = rng.random()
        self.w5 = rng.random()
        self.w6 = rng.random()
        # biases
        self.b1 = rng.random()
        self.b2 = rng.random()
        self.b3 = rng.random()
    
    def feedforward(self, x):
        # x is a array with 2 elements
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
    
    def train(self, data, y_trues):
        learn_rate = 0.1
        epochs = 1000
        loss_list = []
        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                
                # Calculate partial derivatives.
                d_l_d_ypred = -2 * (y_true - y_pred)
                # neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                # neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                # neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                
                # Update weights and biaes.
                # neuron h1
                self.w1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                # neuron h2
                self.w3 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                # neuron o1
                self.w5 -= learn_rate * d_l_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_l_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_l_d_ypred * d_ypred_d_b3
                
            if epoch % 10 == 0:
                y_preds = numpy.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(y_trues, y_preds)
                loss_list.append(loss)
        
        pyplot.plot(range(1, len(loss_list) + 1), loss_list)
        pyplot.grid(True)
        pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
        pyplot.show()

if __name__ == '__main__':
    data = numpy.array([[-2, -1], [25, 6], [17, 4], [-15, -6],])
    y_trues = numpy.array([1, 0, 0, 1])
    network = OurNeuralNetwork()
    network.train(data, y_trues)
    
    emily = numpy.array([-7, -3])
    frank = numpy.array([20, 2])
    print('Emily: %.3f' %network.feedforward(emily))
    print('Frank: %.3f' % network.feedforward(frank))
