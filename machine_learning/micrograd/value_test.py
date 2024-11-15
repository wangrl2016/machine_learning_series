import math
import unittest
import value
import torch

class ValueTest(unittest.TestCase):
    def setUp(self):
        self.a = value.Value(3.0)
        self.b = value.Value(4.0)
    
    def test_add(self):
        a = torch.tensor(3.0, requires_grad=True)
        b = torch.tensor(4.0, requires_grad=True)
        c = a + b
        c.backward()
        print('dc_da:', a.grad)
        print('dc_db:', b.grad)

        c = self.a + self.b
        c.backward()
        assert math.isclose(a.grad.item(), self.a.grad)
        assert math.isclose(b.grad.item(), self.b.grad)

if __name__ == '__main__':
    unittest.main()
