import jax.numpy as jnp
from jax import grad
from jax import jit

# SELU activation function
def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

if __name__ == '__main__':
    # JAX as NumPy
    x = jnp.arange(5.0)
    print(selu(x))

    # Just-in-time compilation with jax.jit()
    selu_jit = jit(selu)
    _ = selu_jit(x) # compiles on first call
    selu_jit(x).block_until_ready()
