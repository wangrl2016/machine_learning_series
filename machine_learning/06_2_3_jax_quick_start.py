import jax.numpy as jnp
from jax import grad

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

if __name__ == '__main__':
    x = jnp.arange(5.0)
    print(selu(x))

    x_small = jnp.arange(3.0)
    derivative_fn = grad(sum_logistic)
    print(derivative_fn(x_small))
