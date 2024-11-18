import math
import unittest
import value
import torch

class ValueTest(unittest.TestCase):
    def setUp(self):
        self.a = value.Value(3.0)
        self.b = value.Value(4.0)
        self.c = value.Value(5.0)
        self.d = value.Value(6.0)
        
        self.ta = torch.tensor(3.0, requires_grad=True)
        self.tb = torch.tensor(4.0, requires_grad=True)
        self.tc = torch.tensor(5.0, requires_grad=True)
        self.td = torch.tensor(6.0, requires_grad=True)
    def test_add(self):
        to = self.ta + self.tb
        to.backward()
        # print('dto_dta:', self.ta.grad)
        # print('dto_dtb:', self.tb.grad)

        o = self.a + self.b
        o.backward()
        assert math.isclose(self.ta.grad.item(), self.a.grad)
        assert math.isclose(self.tb.grad.item(), self.b.grad)
    
    def test_mul(self):
        to = self.tc * (self.ta + self.tb)
        to.backward()
        # print('dto_dtc:', self.tc.grad)

        o = self.c * (self.a + self.b)
        o.backward()
        assert math.isclose(self.tc.grad.item(), self.c.grad)
        assert math.isclose(self.tb.grad.item(), self.b.grad)
        assert math.isclose(self.ta.grad.item(), self.a.grad)
    
    def test_pow(self):
        to = self.td ** (self.tc * (self.tb - self.ta))
        to.backward()
        # print('dto_dtd:', self.td.grad)

        o = self.d ** (self.c * (self.b - self.a)).value
        o.backward()
        assert math.isclose(self.td.grad.item(), self.d.grad)

if __name__ == '__main__':
    unittest.main()
