class Value:
    # stores a single scalar value and its gradient
    def __init__(self, value, _parent=(), _operation=''):
        self.value = value
        self.grad = 0
        # internal variables used for autograd graph construction
        self._gradient_func = lambda: None
        self._parent_node = set(_parent)
        # the operation that produced this node, for graphviz / debugging / etc
        self._operation = _operation

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), '+')

        # z = x + y
        # dz_dx = 1
        # dz_dy = 1
        def gradient_func():
            self.grad += 1
            other.grad += 1
        out._gradient_func = gradient_func

        return out
    