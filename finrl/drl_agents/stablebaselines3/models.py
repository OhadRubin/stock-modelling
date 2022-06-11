# DRL models from Stable Baselines 3

import time

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl import config


MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
            self,
            model_name,
            policy="MlpPolicy",
            policy_kwargs=None,
            model_kwargs=None,
            verbose=1,
            seed=None,
            tensorboard_log=None,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )

    def train_model(self, model, tb_log_name, total_timesteps=5000):
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        return model


    @staticmethod
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True, status=None, live=False, status_bool=False):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            if config.comments_bool:
                print("Successfully load model", cwd)
        except BaseException:
            raise ValueError("Fail to load agent!")

        config.current_model_name = model_name
        if config.comments_bool:
            print("Initial total assets with the agent is ", environment.initial_total_asset)
        # test on the testing env
        if live:
            state = environment.live_reset()
        else:
            state = environment.reset()
        if config.comments_bool:
            # comment out later from
            print("Following is the initial state given as input to the agent while testing ***********")
            # print("type of the state variable is ", type(state))
            print()
            print(state)
            print('***********************************************************')
            # comment out later to

        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [environment.initial_total_asset]
        done = False
        while not done:
            if config.comments_bool:
                print("State given as input for current step is ", state)
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, step_status = environment.step(action)
            if 'transaction_date' not in config.action_info:
                config.action_info['transaction_date'] = []
            config.action_info['transaction_date']
            total_asset = (
                    environment.amount
                    + (environment.price_ary[environment.day] * environment.stocks).sum()
            )


            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_total_asset

            episode_returns.append(episode_return)

            if config.comments_bool:
                print(
                    f"total asset (balance {environment.amount} + stock value) after the current action is {round(total_asset, 2)} ", )
                print("return ratio from action performed in current time step is ", round(episode_return, 3), "%")
                print("new state after the action performed is ", state)
                print("###############################################################################################")
                print()
        print(f"return ratio (total portfolio val / initial portfolio val) while using {model_name} is {round(episode_return, 2)}")
        print("Test Finished!")
        if status_bool:
            return episode_total_assets, step_status
        else:
            return episode_total_assets


