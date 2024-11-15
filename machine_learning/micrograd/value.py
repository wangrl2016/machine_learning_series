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
            self.grad = 1.0
            other.grad = 1.0
        out._backward = gradient
        return out

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
        
        # go one variable at a time and apply the chain rule to get its gradient
        for v in reversed(topo):
            v._backward()

    