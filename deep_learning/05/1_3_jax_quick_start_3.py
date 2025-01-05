from jax import random, vmap, jit
import jax.numpy as jnp
import numpy

def scalar_function(x):
    return x**2


if __name__ == '__main__':
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    print(scalar_function(x))
    batched_function = vmap(scalar_function)
    print(batched_function(x))

    key = random.key(1701)
    key1, key2 = random.split(key)
    mat = random.normal(key1, (150, 100))
    print(mat.shape)
    batched_x = random.normal(key2, (10, 100))
    print(batched_x.shape)

    def apply_matrix(x):
        return jnp.dot(mat, x)
    
    def naively_batched_apply_matrix(v_batched):
        return jnp.stack([apply_matrix(v) for v in v_batched])
    
    @jit
    def batched_apply_matrix(batched_x):
        return jnp.dot(batched_x, mat.T)
    
    numpy.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                                  batched_apply_matrix(batched_x),
                                  atol=1e-4, rtol=1e-4)
    batched_apply_matrix(batched_x).block_until_ready()

    @jit
    def vmap_batched_apply_matrix(batched_x):
        return vmap(apply_matrix)(batched_x)

    numpy.testing.assert_allclose(naively_batched_apply_matrix(batched_x),
                                  vmap_batched_apply_matrix(batched_x),
                                  atol=1e-4, rtol=1e-4)
    vmap_batched_apply_matrix(batched_x).block_until_ready()
        