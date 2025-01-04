import numpy

def kernel_func(query, key):
    return 1 / (query - key)**2

def softmax(arr):
    sum = 0
    result = []
    for a in arr:
        sum += numpy.power(numpy.e, a)
    for a in arr:
        result.append(numpy.power(numpy.e, a) / sum)
    return result

if __name__ == '__main__':
    key_value_dict = {
        1: 2, 2:4, 3: 8, 4:16, 5:32,
    }
    query = 2.8

    attention_weights = []
    for key, value in key_value_dict.items():
        attention_weights.append(round(kernel_func(query, key), 3))
    print(attention_weights)

    probabilities = numpy.round(softmax(attention_weights), 3)
    print(probabilities)
