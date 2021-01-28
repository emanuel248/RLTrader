'''

A large part of the code in this file was sourced from the rl-baselines-zoo library on GitHub.
In particular, the library provides a great parameter optimization set for the PPO2 algorithm,
as well as a great example implementation using optuna.

Source: https://github.com/araffin/rl-baselines-zoo/blob/master/utils/hyperparams_opt.py

'''

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import optuna

import pandas as pd
import numpy as np
import math

import os

from datetime import datetime

from tqdm import tqdm, trange

from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import PPO2, A2C

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# number of parallel jobs
n_jobs = 2
# maximum number of trials for finding the best hyperparams
n_trials = 100
# number of test episodes per trial
n_test_episodes = 3
# number of evaluations for pruning per trial
n_evaluations = 6

n_envs = 9


def make_env(data_frame, env_params, rank, seed=0):
    def _init():
        env = BitcoinTradingEnv(data_frame,  **env_params)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init

def optimize_envs(trial):
    return {
        'reward_func': 'sortino',
        'forecast_len': int(trial.suggest_loguniform('forecast_len', 1, 200)),
        'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
    }


def optimize_ppo2(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }


def optimize_agent(trial):
    print('Preparing env..')
    env_params = optimize_envs(trial)

    train_env = DummyVecEnv(
        [make_env(train_df, env_params, i) for i in range(n_envs)])
    #train_env = DummyVecEnv(
    #    [lambda: BitcoinTradingEnv(train_df,  **env_params)])
    test_env = SubprocVecEnv(
        [make_env(test_df, env_params, i) for i in range(n_envs)])
    #test_env = DummyVecEnv(
    #    [lambda: BitcoinTradingEnv(test_df, **env_params)])

    print('..init model..')
    model_params = optimize_ppo2(trial)
    model = PPO2(MlpLnLstmPolicy, train_env, verbose=0, nminibatches=3,
                 tensorboard_log=None, **model_params)

    last_reward = -np.inf
    evaluation_interval = int(len(train_df) / n_evaluations)

    print('..env ready')
    for eval_idx in range(n_evaluations):
        try:
            print('start training ', datetime.now())
            model.learn(evaluation_interval)
            print('end training ', datetime.now())
        except AssertionError:
            raise

        rewards = []
        n_episodes, reward_sum = 0, 0.0

        # trades = train_env.get_attr('trades')
        # if len(trades) < 1:
        #    raise optuna.structs.TrialPruned()

        obs = test_env.reset()
        step_cnt = 0
        while n_episodes < n_test_episodes:
            action, _ = model.predict(obs)
            obs, reward, done, _ = test_env.step(action)
            reward_sum += sum(reward)

            if step_cnt % 16 == 0:
                try:
                    test_env.render(mode='human')
                except:
                    pass

            if all(done):
                rewards.append(reward_sum)
                reward_sum = 0.0
                n_episodes += 1
                obs = test_env.reset()
            step_cnt = step_cnt + 1

            trial.report(-1 * last_reward, step_cnt)
            if trial.should_prune():
                raise optuna.structs.TrialPruned()

        last_reward = np.mean(rewards)


    return -1 * last_reward


def optimize():
    study = optuna.create_study(
        study_name='ppo2_sortino', storage='sqlite:///params.db', load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=32, interval_steps=16))

    try:
        study.optimize(optimize_agent, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


if __name__ == '__main__':
    print('Loading dataset..')
    df = pd.read_csv('./data/coinbase_hourly.csv')
    df = df.drop(['Symbol'], axis=1)
    df = df.sort_values(['Date'])
    df = add_indicators(df.reset_index())
    print('..done')

    train_len = int(len(df) - len(df) * 0.1)
    train_df = df[:train_len]
    test_df = df[train_len:]

    optimize()
