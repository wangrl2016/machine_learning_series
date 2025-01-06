import torch
import numpy

if __name__ == '__main__':
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    
    np_array = numpy.array(data)
    x_np = torch.from_numpy(np_array)
    
    x_ones = torch.ones_like(x_data)
    print('Ones tensor:', x_ones)
    
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print('Random tensor:', x_rand)

    shape = (2, 3, )
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print('Random tensor:\n', rand_tensor)
    print('Ones tensor:\n', ones_tensor)
    print('Zeros tensor:\n', zeros_tensor)
    
    tensor = torch.rand(3, 4)
    print('Shape of tensor:', tensor.shape)
    print('Datatype of tensor:', tensor.dtype)
    print('Device tensor is stored on:', tensor.device)