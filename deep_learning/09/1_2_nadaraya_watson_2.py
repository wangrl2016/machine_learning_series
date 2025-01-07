import numpy
from matplotlib import pyplot

def kernel_standard_normal(u):
    return (1 / numpy.sqrt(2*numpy.pi)) * numpy.exp(-0.5*u**2)

def nadaraya_watson(x_train, y_train, x_query, h, kernel=kernel_standard_normal):
    y_pred = []
    for x in x_query:
        weights = kernel((x - x_train) / h)
        numerator = numpy.sum(weights * y_train)
        denominator = numpy.sum(weights)
        y_pred.append(numerator / denominator)
    return numpy.array(y_pred)

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    x_train = numpy.linspace(-3, 3, 60)
    y_train = numpy.sin(x_train) + rng.normal(0, 0.3, x_train.shape)
    x_query = numpy.linspace(-3, 3, 100)
    h = 0.5
    y_pred = nadaraya_watson(x_train, y_train, x_query, h)
    
    pyplot.scatter(x_train, y_train, color='blue', label='Train data')
    pyplot.plot(x_query, y_pred, color='red', label='Nadaraya-Watson regression')
    pyplot.legend()
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
