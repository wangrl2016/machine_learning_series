class Value:
    # stores a single scalar value and its gradient
    def __init__(self, value, _parent=(), _operation=''):
        self.value = value
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._parent_nodes = set(_parent)
        # the operation that produced this node, for graphviz / debugging / etc
        self._operation = _operation

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), '+')
        # z = x + y
        # dz_dx = 1
        # dz_dy = 1
        def gradient():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = gradient
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other), '*')
        # z = x * y
        # dz_dx = y
        # dz_dy = x
        def gradient():
            self.grad = other.value * out.grad
            other.grad = self.value * out.grad
        out._backward = gradient
        return out

    def __neg__(self): # -self
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only supporting int or float powers for now'
        out = Value(self.value**other, (self,), f'^{other}')
        # y = x^n (n 是常数)
        # dy_dx = n * x^(n - 1)
        def gradient():
            self.grad = (other * self.value**(other - 1)) * out.grad
        out._backward = gradient
        return out

    def __truediv__(self, other):
        return self * other**-1

    def relu(self):
        out = Value(0 if self.value < 0 else self.value, (self,), 'ReLU')
        def gradient():
            self.grad = (out.value > 0) * out.grad
        out._backward = gradient

    def backward(self):
        # topological order all of the parent in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for node in v._parent_nodes:
                    build_topo(node)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        # go one variable at a time and apply the chain rule to get its gradient
        for v in reversed(topo):
            v._backward()
