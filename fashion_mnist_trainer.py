import jax.numpy as jnp
from jax import random, grad, jit, Array
from fashion_mnist import load_fashion_mnist_dataset


Parameters = list[tuple[Array, Array]]


def initialise_params(n_neurons: list[int]) -> Parameters:
    random_keys = random.split(random.key(0), len(n_neurons))
    scale_factor: float = 0.01
    def initialise_single_params(in_neuron: int, out_neuron: int, key: "random.key") -> tuple[Array, Array]:
        w_key, b_key = random.split(key)
        return scale_factor * random.normal(w_key, (in_neuron, out_neuron)), \
                scale_factor* random.normal(b_key, (out_neuron,))

    return [initialise_single_params(n, m, k) for n, m, k in zip(n_neurons, n_neurons[1:], random_keys)]


def linear_forward_pass(params: Parameters, x: Array) -> Array:
    activations = x
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = jnp.maximum(0.0, outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits


def mse_loss(params: Parameters, x: Array, y: Array) -> Array:
    out = linear_forward_pass(params, x)
    return jnp.power(y - out, 2).mean(0).sum()


@jit
def update_gradients(params: Parameters, x: Array, y: Array, learning_rate: float):
    loss = mse_loss(params, x, y)
    grads = grad(mse_loss)(params, x, y)
    return loss, [(w - learning_rate * dw, b - learning_rate * db) for (w, b), (dw, db) in zip(params, grads)]


if __name__ == "__main__":

    EPOCHS = 50
    LEARNING_RATE = 0.01

    # Load Fashion MNIST dataset
    x_train, y_train = load_fashion_mnist_dataset()

    # Split into batches
    BS = 2048
    x_train = [x_train[BS*i:BS*(i+1)] for i in range(x_train.shape[0]//BS)]
    y_train = [y_train[BS*i:BS*(i+1)] for i in range(y_train.shape[0]//BS)]

    # Initialise parameters
    params = initialise_params([784, 512, 128, 64, 10])

    # Train
    for epoch in range(EPOCHS):
        losses = None
        for x_batch, y_batch in zip(x_train, y_train):
            loss, params = update_gradients(params, x_batch, y_batch, LEARNING_RATE)
            losses = jnp.expand_dims(loss, 0) if losses is None else jnp.concatenate([losses, jnp.expand_dims(loss, 0)])
        mean_epoch_loss = losses.mean()
        print(f"{epoch=} {mean_epoch_loss=}")

    # Test
    predictions = linear_forward_pass(params, jnp.concatenate(x_train))
    print(f"Accuracy: {(jnp.mean(jnp.argmax(predictions, axis=1) == jnp.argmax(jnp.concatenate(y_train), axis=1)))}")
