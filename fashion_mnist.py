import gzip
import requests

import jax.numpy as jnp
from jax import Array


def load_fashion_mnist_dataset() -> tuple[Array, Array]:
    images_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
    labels_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"

    images = jnp.frombuffer(gzip.decompress(requests.get(images_url, timeout=30).content), dtype=jnp.uint8, offset=16)
    labels = jnp.frombuffer(gzip.decompress(requests.get(labels_url, timeout=30).content), dtype=jnp.uint8, offset=8)
    images = jnp.reshape(images, (60000, 784)).astype(jnp.float32)
    labels = jnp.array(labels[:, None] == jnp.arange(10), jnp.float32)
    return images, labels
