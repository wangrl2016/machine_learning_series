import jax.numpy as jnp
from jax import grad, jit, jacobian

def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

def first_finite_differences(f, x, eps=1E-3):
    return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps)
                   for v in jnp.eye(len(x))])

if __name__ == '__main__':
    # Taking derivatives with jax.grad()
    x_small = jnp.arange(3.0)
    derivative_fn = grad(sum_logistic)
    print(derivative_fn(x_small))
    print(first_finite_differences(sum_logistic, x_small))
    assert jnp.allclose(first_finite_differences(sum_logistic, x_small),
        derivative_fn(x_small), 1e-3)

    print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

    print(jacobian(jnp.exp)(x_small))
