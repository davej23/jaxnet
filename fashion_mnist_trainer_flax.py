import optax
import jax.numpy as jnp
import flax.linen as nn

from flax.training import train_state
from jax import random, value_and_grad, jit, Array
from fashion_mnist import load_fashion_mnist_dataset


class SimpleFlaxModel(nn.Module):
    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(512)(x)
        # x = nn.relu(x)
        x = nn.Dense(128)(x)
        # x = nn.relu(x)
        x = nn.Dense(64)(x)
        # x = nn.relu(x)
        x = nn.Dense(10)(x)
        # x = nn.softmax(x)
        return x


@jit
def apply_model(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn(params, x)
        # loss = optax.squared_error(logits, y).mean()
        loss = jnp.power(y - logits, 2).mean(0).sum()
        return loss, logits
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    acc = jnp.mean(jnp.argmax(nn.softmax(logits), axis=1) == jnp.argmax(y, axis=1))
    return state, grads, loss, acc

@jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


if __name__ == "__main__":

    EPOCHS = 50
    LEARNING_RATE = 0.01

    # Load Fashion MNIST dataset
    x_train, y_train = load_fashion_mnist_dataset()

    # Split into batches
    BS = 256
    x_train = [x_train[BS*i:BS*(i+1)] for i in range(x_train.shape[0]//BS)]
    y_train = [y_train[BS*i:BS*(i+1)] for i in range(y_train.shape[0]//BS)]

    # Initialise parameters
    model = SimpleFlaxModel()
    params = model.init(random.key(0), jnp.zeros((1, 784)))
    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # Train
    for epoch in range(EPOCHS):
        losses = None
        accs = None
        for x_batch, y_batch in zip(x_train, y_train):
            state, grads, loss, acc = apply_model(state, x_batch, y_batch)
            state = update_model(state, grads)
            losses = jnp.expand_dims(loss, 0) if losses is None else jnp.concatenate([losses, jnp.expand_dims(loss, 0)])
            accs = jnp.expand_dims(acc, 0) if accs is None else jnp.concatenate([accs, jnp.expand_dims(acc, 0)])

        mean_epoch_loss = losses.mean()
        mean_accuracy = accs.mean()
        print(f"{epoch=} {mean_epoch_loss=}, {mean_accuracy=}")

    # Test
    _, _, loss, acc = apply_model(state, jnp.concatenate(x_train), jnp.concatenate(y_train))
    print(f"test set --- {loss=} {acc=}")
    print(nn.softmax(state.apply_fn(params, jnp.concatenate(x_train))[:10]))
    print(jnp.concatenate(y_train)[:10])