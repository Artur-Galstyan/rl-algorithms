import pathlib
from typing import List
from tqdm import tqdm
import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import tensorflow_probability.substrates.jax as tfp
from jaxtyping import Array, PyTree
from little_helpers.equinox_helpers import eqx_init_optimiser
from little_helpers.rl_helpers import get_future_rewards
from little_helpers.gym_helpers import get_trajectory
from optax import GradientTransformation
import matplotlib.pyplot as plt
import logging

_logger = logging.getLogger(__name__)


def loss_fn(
    model: eqx.Module,
    states: Array,
    rewards: Array,
    actions: Array,
) -> Array:
    logits = jax.vmap(model)(states)
    advantages = get_future_rewards(rewards)

    actions = actions.reshape(-1)

    loss_value = rlax.policy_gradient_loss(
        logits_t=logits,
        a_t=actions,
        adv_t=advantages,
        w_t=jnp.ones_like(advantages),
    )

    return loss_value


def act(obs: int, model: eqx.Module, key: jax.random.PRNGKeyArray) -> int:
    key, subkey = jax.random.split(key)
    logits = model(obs)
    action = tfp.distributions.Categorical(logits=logits).sample(seed=subkey)
    return action


@eqx.filter_jit
def step(
    model: eqx.Module,
    eps_obs: Array,
    rewards: List,
    eps_actions: Array,
    optim: GradientTransformation,
    opt_state: PyTree,
):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
        model, eps_obs, rewards, eps_actions
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


class Model(eqx.Module):
    layers: list

    def __init__(self, n_in, n_out, key: jax.random.PRNGKeyArray):
        super().__init__()

        key1, key2, key3 = jax.random.split(key, 3)

        self.layers = [
            # eqx.nn.Embedding(n_in, 4, key=key1),
            eqx.nn.Linear(n_in, 4, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(4, 6, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(6, n_out, key=key3),
        ]

    def __call__(self, x: int) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    n_demo_episodes = 3
    learning_rate = 0.001
    n_epochs = 5
    n_episodes = 500
    max_eps_steps = 200

    # env_name = "CliffWalking-v0"
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_eps_steps)
    obs, info = env.reset()

    # obs_space = env.observation_space.n
    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.n

    key = jax.random.PRNGKey(0)
    model = Model(obs_space, act_space, key)

    optim = optax.adamw(learning_rate=learning_rate)
    opt_state = eqx_init_optimiser(optim, model)

    epoch_reward = []
    for epoch in range(n_epochs):
        total_rewards = []
        for episode in range(n_episodes):
            key, subkey = jax.random.split(key)
            rewards, eps_obs, eps_actions = get_trajectory(
                env=env,
                key=subkey,
                act_fn=act,
                act_fn_kwargs={"model": model},
            )
            model, opt_state, loss_value = step(
                model, eps_obs, rewards, eps_actions, optim, opt_state
            )
            total_rewards.append(np.sum(rewards))
        epoch_reward.append(np.mean(total_rewards))
        _logger.info(f"Epoch: {epoch}, Mean Reward: {np.mean(total_rewards)}")
    # plt.plot(epoch_reward)
    # plt.show()

    # Saving the model
    model_path = pathlib.Path(__file__).parent / "models/vanilla_policy_gradient"
    _logger.info(
        f"Saving model to: {model_path}/{env_name}-{np.round(np.mean(epoch_reward), 2)}-model.eqx"  # noqa: E501
    )

    eqx.tree_serialise_leaves(
        f"{model_path}/{env_name}-{np.round(np.mean(epoch_reward), 2)}-model.eqx",
        model,
    )
    """
    for episode in range(n_demo_episodes):
        rewards, obs, actions = get_trajectory(
            env=env,
            key=key,
            act_fn=act,
            act_fn_kwargs={"model": model},
            render=True,
        )
        print(f"Episode: {episode}, Reward: {np.sum(rewards)}")
    """


if __name__ == "__main__":
    main()
