import logging
import pathlib
import sys
from typing import List
from functools import partial
import haiku as hk

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import rlax
import tensorflow_probability.substrates.jax as tfp
import torch
from gymnasium import Env
from jaxtyping import Array, PyTree
from little_helpers.equinox_helpers import eqx_init_optimiser
from little_helpers.gym_helpers import get_trajectory

# from little_helpers.rl_helpers import get_future_rewards
from optax import GradientTransformation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_logger = logging.getLogger(__name__)


class HaikuModel(hk.Module):
    def __init__(self, n_in: int, n_out: int, h1: int, h2: int):
        super().__init__()
        self.n_out = n_out
        self.h1 = h1
        self.h2 = h2

    def __call__(self, x: int) -> Array:
        x = hk.Linear(self.h1)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.h2)(x)
        x = jax.nn.relu(x)
        x = hk.Linear(self.n_out)(x)
        return x


# @jax.jit
def get_future_rewards(rewards: jnp.ndarray, gamma=0.99) -> jnp.ndarray:
    returns = jnp.zeros_like(rewards)
    future_returns = 0

    for t in range(len(rewards) - 1, -1, -1):
        future_returns = rewards[t] + gamma * future_returns
        returns = returns.at[t].set(future_returns)

    return returns


"""
def get_future_rewards(rewards: jnp.ndarray, gamma=0.99) -> jnp.ndarray:
    def scan_fn(carry, reward):
        return (carry * gamma,) + reward, None

    _, returns = jax.lax.scan(scan_fn, 0, rewards[::-1])
    return returns[::-1]


def get_future_rewards(rewards: Array, gamma=0.99) -> Array: 
    print(f"get_future_rewards_fn: {type(rewards)=}")
    T = len(rewards)
    returns = np.empty(T)
    future_returns = 0.0
    for t in reversed(range(T)):
        future_returns = rewards[t] + gamma * future_returns
        returns[t] = future_returns
    return jnp.array(returns)
"""


class RLDataset(Dataset):
    def __init__(self, rewards, actions, obs):
        self.rewards = rewards
        self.actions = actions
        self.obs = obs

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, index):
        reward = torch.tensor(self.rewards[index])
        action = torch.tensor(self.actions[index])
        ob = torch.tensor(self.obs[index])
        return ob, action, reward


# @partial(jax.jit, static_argnums=(1,))
def act(obs: int, model: hk.Module, key: jax.random.PRNGKeyArray, params) -> int:
    key, subkey = jax.random.split(key)
    logits = model.apply(params, None, obs)
    action = tfp.distributions.Categorical(logits=logits).sample(seed=subkey)
    return action


def train(
    t_model: hk.Module,
    t_params: PyTree,
    t_optim: GradientTransformation,
    t_opt_state: PyTree,
    t_eps_obs: Array,
    t_eps_actions: Array,
    t_eps_rewards: Array,
):
    # @jax.jit
    def loss_fn(
        l_params: optax.Params,
        l_states: Array,
        l_rewards: Array,
        l_actions: Array,
    ) -> Array:
        logits = t_model.apply(l_params, None, l_states)
        advantages = get_future_rewards(l_rewards)

        actions = l_actions.reshape(-1)

        l_loss_value = rlax.policy_gradient_loss(
            logits_t=logits,
            a_t=actions,
            adv_t=advantages,
            w_t=jnp.ones_like(advantages),
        )

        return l_loss_value

    # @jax.jit
    def step(
        s_params: optax.Params,
        s_eps_obs: Array,
        s_eps_rewards: List,
        s_eps_actions: Array,
        s_opt_state: PyTree,
    ):
        s_loss_value, grads = jax.value_and_grad(loss_fn)(
            s_params, s_eps_obs, s_eps_rewards, s_eps_actions
        )
        updates, s_opt_state = t_optim.update(grads, s_opt_state, s_params)
        s_params = optax.apply_updates(s_params, updates)
        return s_params, s_opt_state, s_loss_value

    params, opt_state, loss_value = step(
        t_params, t_eps_obs, t_eps_rewards, t_eps_actions, t_opt_state
    )
    return params, opt_state, loss_value


def experiment(env: Env, hyperparams: dict) -> dict:
    _logger.info(f"Running experiment on: {env.spec.name}")
    env_name = env.spec.name
    obs, info = env.reset()
    # Hyperparameters
    learning_rate = hyperparams["learning_rate"]
    n_epochs = hyperparams["n_epochs"]
    n_episodes = hyperparams["n_episodes"]

    h1 = hyperparams["h1"]
    h2 = hyperparams["h2"]

    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.n

    model = hk.transform(lambda x: HaikuModel(obs_space, act_space, h1, h2)(x))
    rng_seq = hk.PRNGSequence(42)
    params = model.init(next(rng_seq), obs)

    optim = optax.adamw(learning_rate=learning_rate)
    opt_state = optim.init(params)
    key = jax.random.PRNGKey(42)
    epoch_reward = []
    for epoch in range(n_epochs):
        total_rewards = []
        for episode in tqdm(range(n_episodes)):
            key, subkey = jax.random.split(key)
            rewards, eps_obs, eps_actions = get_trajectory(
                env=env,
                key=subkey,
                act_fn=act,
                act_fn_kwargs={"model": model, "params": params},
            )
            dataset = RLDataset(
                np.array(rewards), np.array(eps_actions), np.array(eps_obs)
            )
            dataloader = DataLoader(
                dataset, batch_size=64, shuffle=False, drop_last=True
            )
            for batch in dataloader:
                obs_batch, action_batch, reward_batch = batch
                params, opt_state, loss_value = train(
                    model,
                    params,
                    optim,
                    opt_state,
                    obs_batch.numpy(),
                    action_batch.numpy(),
                    reward_batch.numpy(),
                )
            total_rewards.append(np.sum(rewards))
        epoch_reward.append(np.mean(total_rewards))
        _logger.info(f"Epoch: {epoch}, Mean Reward: {np.mean(total_rewards)}")
    plt.plot(epoch_reward)
    plt.title(f"Mean Reward per Epoch - {env_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Reward")
    plot_path = pathlib.Path(__file__).parent / "plots/vanilla_policy_gradient"
    plt.savefig(f"{plot_path}/{env_name}.png")

    # Saving the model
    model_path = pathlib.Path(__file__).parent / "models/vanilla_policy_gradient"
    _logger.info(
        f"Saving model to: {model_path}/{env_name}-{np.round(np.mean(epoch_reward), 2)}-model.eqx"  # noqa: E501
    )

    eqx.tree_serialise_leaves(
        f"{model_path}/{env_name}-{np.round(np.mean(epoch_reward), 2)}-model.eqx",
        model,
    )
    return {
        "model": model,
        "epoch_reward": epoch_reward,
        "total_rewards": total_rewards,
    }


def main():
    max_eps_steps = 200

    env_names = {
        # "CartPole-v1": 200,
        # "MountainCar-v0": None,
        # "Acrobot-v1": None,
        "LunarLander-v2": None,
    }

    envs = []

    for env_name, max_eps_steps in env_names.items():
        env = gym.make(env_name)
        if max_eps_steps:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_eps_steps)
        envs.append(env)

    hyperparams = {
        "learning_rate": 0.001,
        "n_epochs": 5,
        "n_episodes": 500,
        "h1": 4,
        "h2": 6,
    }

    for env in envs:
        env_name = env.spec.name
        _experiment = experiment(env, hyperparams)
        _logger.info(f"Experiment Complete: {env_name}")
        _logger.info(f"Mean Reward: {np.mean(_experiment['epoch_reward'])}")
        _logger.info(f"Std Reward: {np.std(_experiment['total_rewards'])}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
