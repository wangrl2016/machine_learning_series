import jax
import jax.numpy

def func(x):
    return x**2

if __name__ == '__main__':
    x = 3.0
    dy_dx = jax.grad(func)(x)
    print(dy_dx)

    w = jax.numpy.full((3, 2), 0.1)
    b = jax.numpy.zeros(2)
    x = jax.numpy.array([[1.0, 2.0, 3.0]])
    def forward(x, w, b):
        y = jax.numpy.tanh(jax.numpy.dot(x, w) + b)
        return jax.numpy.mean(y * y)
    grads = jax.grad(forward, argnums=(1, 2))(x, w, b)
    dl_dw, dl_db = grads
    print(dl_dw)
    print(dl_db)
