import logging
import pathlib
from typing import List

import optuna

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

from little_helpers.rl_helpers import get_future_rewards
from optax import GradientTransformation
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_logger = logging.getLogger(__name__)


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


@eqx.filter_jit
def act(obs: int, model: eqx.Module, key: jax.random.PRNGKeyArray) -> int:
    key, subkey = jax.random.split(key)
    logits = model(obs)
    action = tfp.distributions.Categorical(logits=logits).sample(seed=subkey)
    return action


class Model(eqx.Module):
    layers: list

    def __init__(
        self, n_in: int, n_out: int, h1: int, h2: int, key: jax.random.PRNGKeyArray
    ):
        super().__init__()

        key1, key2, key3 = jax.random.split(key, 3)

        self.layers = [
            # eqx.nn.Embedding(n_in, 4, key=key1),
            eqx.nn.Linear(n_in, h1, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(h1, h2, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(h2, n_out, key=key3),
        ]

    def __call__(self, x: int) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


@eqx.filter_jit
def train(
    t_optim: GradientTransformation,
    t_opt_state: PyTree,
    t_params: PyTree,
    t_eps_obs: Array,
    t_eps_actions: Array,
    t_eps_rewards: Array,
):
    def loss_fn(
        l_model: PyTree,
        l_states: Array,
        l_rewards: Array,
        l_actions: Array,
    ) -> Array:
        logits = jax.vmap(l_model)(l_states)
        advantages = get_future_rewards(l_rewards)

        actions = l_actions.reshape(-1)

        l_loss_value = rlax.policy_gradient_loss(
            logits_t=logits,
            a_t=actions,
            adv_t=advantages,
            w_t=jnp.ones_like(advantages),
        )

        return l_loss_value

    def step(
        s_params: PyTree,
        s_eps_obs: Array,
        s_eps_rewards: List,
        s_eps_actions: Array,
        s_opt_state: PyTree,
    ):
        s_loss_value, grads = eqx.filter_value_and_grad(loss_fn)(
            s_params, s_eps_obs, s_eps_rewards, s_eps_actions
        )
        updates, s_opt_state = t_optim.update(grads, s_opt_state, s_params)
        s_params = eqx.apply_updates(s_params, updates)
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

    key = jax.random.PRNGKey(0)

    model = Model(obs_space, act_space, h1, h2, key)

    optim = optax.adamw(learning_rate=learning_rate)
    opt_state = eqx_init_optimiser(optim, model)

    epoch_reward = []
    for epoch in range(n_epochs):
        total_rewards = []
        for episode in tqdm(range(n_episodes)):
            key, subkey = jax.random.split(key)
            rewards, eps_obs, eps_actions = get_trajectory(
                env=env,
                key=subkey,
                act_fn=act,
                act_fn_kwargs={"model": model},
            )

            dataset = RLDataset(
                np.array(rewards), np.array(eps_actions), np.array(eps_obs)
            )
            dataloader = DataLoader(
                dataset, batch_size=64, shuffle=False, drop_last=True
            )
            for batch in dataloader:
                obs_batch, action_batch, reward_batch = batch
                model, opt_state, loss_value = train(
                    optim,
                    opt_state,
                    model,
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
    plt.savefig(f"{plot_path}/{env_name}-{np.round(np.mean(epoch_reward), 2)}.png")

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


def objective(trial: optuna.Trial, env_name: str, max_eps_steps: int) -> float:
    env = gym.make(env_name)
    if max_eps_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_eps_steps)

    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
        "n_epochs": 10,
        "n_episodes": 1000,
        "h1": trial.suggest_int("h1", 1, 20),
        "h2": trial.suggest_int("h2", 1, 20),
    }

    _experiment = experiment(env, hyperparams)
    mean_reward = np.mean(_experiment["epoch_reward"])
    return mean_reward


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    env_names = [
        # "CartPole-v1",
        "MountainCar-v0",
        "Acrobot-v1",
        "LunarLander-v2",
    ]
    max_eps_steps = [200, None, None, None]
    current_file_path = pathlib.Path(__file__).parent.absolute()
    database_url = current_file_path.parent / "studies/studies.db"
    for env_name, max_eps_steps in zip(env_names, max_eps_steps):
        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{database_url}",
            study_name=env_name,
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: objective(trial, env_name, max_eps_steps),
            n_trials=20,
        )
        _logger.info(f"Best trial: {study.best_trial.value}")
        _logger.info(f"Best trial params: {study.best_trial.params}")
