'''

A large part of the code in this file was sourced from the rl-baselines-zoo library on GitHub.
In particular, the library provides a great parameter optimization set for the PPO2 algorithm,
as well as a great example implementation using optuna.

Source: https://github.com/araffin/rl-baselines-zoo/blob/master/utils/hyperparams_opt.py

'''

import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.python.framework.ops import disable_eager_execution

import optuna
import tracemalloc as tr
import multiprocessing

import pandas as pd
import numpy as np
import math

import os
import gc
import cv2

from datetime import datetime
import time

from tqdm import tqdm, trange

from baselines.ppo2 import ppo2
from baselines.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, SubprocVecEnv
from baselines.common import set_global_seeds

from env.BitcoinTradingEnv import BitcoinTradingEnv, DataProvider
from util.indicators import add_indicators

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optimize bitcoin trading bot')
parser.add_argument('study', type=str,
                    help='Name for study')
parser.add_argument('--dataset', type=str, help='Dataset to use (.8/.2)', default='data/coinbase_hourly.csv')
parser.add_argument('--cleft', type=str, help='Currency left side', default='BTC')
parser.add_argument('--cright', type=str, help='Currency right side', default='USD')
parser.add_argument('--balance', type=int,
                    help='Initial', default=10000)
parser.add_argument('--envs', type=int,
                    help='Number of envs to spawn', default=2)
parser.add_argument("--test", type=bool)
parser.add_argument("--memtest", type=bool)
parser.add_argument("--modeltest", type=bool)
args = parser.parse_args()

# number of parallel jobs
n_jobs = 2
# maximum number of trials for finding the best hyperparams
n_trials = 100
# number of test episodes per trial
n_test_episodes = 3
# number of evaluations for pruning per trial
n_evaluations = 4

n_envs = args.envs


def logtime(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    time_diff = time.time() - start_time
    print(f'{func.__name__} ran {time_diff} s')
    return result


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
        'forecast_len': int(trial.suggest_loguniform('forecast_len', 1, 20)),
        'confidence_interval': trial.suggest_uniform('confidence_interval', 0.7, 0.99),
    }


def optimize_ppo2(trial):
    return {
        'nsteps': int(trial.suggest_loguniform('nsteps', 1024, 4096)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'lr': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.),
    }


def optimize_agent(trial):
    start_time = time.time()
    print('Preparing/cleaning env..')

    env_params = optimize_envs(trial)

    train_env = SubprocVecEnv(
        [make_env(train_provider, env_params, i) for i in range(n_envs)])
    test_env = SubprocVecEnv(
        [make_env(test_provider, env_params, i) for i in range(n_envs)])

    time_diff =  time.time() - start_time
    model_params = optimize_ppo2(trial)

    last_reward = -np.inf
    evaluation_interval = int(len(train_df) / n_evaluations)

    time_diff = time.time() - start_time
    print(f'..env ready after {time_diff} s')
    with tqdm(total=n_test_episodes, ascii=True) as ep_bar:
        with tqdm(total=len(test_df), ascii=True) as pbar:
            for episode in trange(n_evaluations):
                ep_bar.set_description_str(desc='training')
                model = ppo2.learn(network='lnlstm', total_timesteps=evaluation_interval, env=train_env, **model_params)
                ep_bar.set_description_str(desc='eval')

                rewards = []
                n_episodes, reward_sum = 0, 0.0

                obs = test_env.reset()
                step_cnt = 0
                
                state = model.initial_state if hasattr(model, 'initial_state') else None
                while n_episodes < n_test_episodes:
                    if state is not None:
                        actions, _, state, _ = model.step(obs)
                    else:
                        actions, _, _, _ = model.step(obs)
                    obs, reward, done, _ = test_env.step(actions.numpy())
                    reward_sum += np.mean(reward)

                    #if step_cnt % 1524 == 0:
                    #    try:
                    #        test_env.render(mode='human')
                    #    except:
                    #        pass

                    if any(done):
                        rewards.append(reward_sum)
                        reward_sum = 0.0
                        n_episodes += 1
                        ep_bar.update(1)
                        pbar.reset()
                        obs = test_env.reset()

                    pbar.update(1)
                    step_cnt = step_cnt + 1
                ep_bar.reset()

                last_reward = np.mean(rewards)
                trial.report(last_reward, episode)
                if trial.should_prune():
                    del model
                    raise optuna.structs.TrialPruned()

    del model
    return last_reward


def optimize():
    study = optuna.create_study(
        study_name=f'ppo2_{args.study}', storage='sqlite:///params_ppo2.db', load_if_exists=True, direction='maximize', 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=32, interval_steps=16))

    try:
        study.optimize(optimize_agent, n_trials=n_trials, n_jobs=1, gc_after_trial=True, show_progress_bar=False)
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
    df = pd.read_csv(args.dataset)
    df = df.drop(['Symbol'], axis=1)
    df = df.sort_values(['Date'])
    df = add_indicators(df.reset_index(), currency=args.cleft)
    print('..done')

    train_len = int(len(df) - len(df) * 0.2)
    train_df = df[:train_len]
    test_df = df[train_len:]


    train_provider = DataProvider(train_df, cur_left=args.cleft, cur_right=args.cright, initial_balance=args.balance)
    test_provider = DataProvider(test_df, cur_left=args.cleft, cur_right=args.cright, initial_balance=args.balance)

    if args.test:
        print('Testing runtime speed..')
        test_env_runtime = BitcoinTradingEnv(test_provider,  {})
        start_time = time.time()
        obs = test_env_runtime.reset()
        for _ in trange(1200):
            obs, reward, done, _ = test_env_runtime.step(0)
        time_diff = time.time() - start_time
        print(f'uncached test ran {time_diff} s')
        start_time = time.time()
        obs = test_env_runtime.reset()
        for _ in trange(1200):
            obs, reward, done, _ = test_env_runtime.step(0)
        time_diff = time.time() - start_time
        print(f'cached test ran {time_diff} s')
    elif args.modeltest:
        start_time = time.time()
        test_env_runtime = DummyVecEnv([make_env(test_provider, {}, i) for i in range(4)])
        ppo2.learn(network='mlp', total_timesteps=1000, env=test_env_runtime)
        time_diff = time.time() - start_time
        print(f'model test ran {time_diff} s')
    else:
        optimize()
