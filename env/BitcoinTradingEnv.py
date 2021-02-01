import gym
import pandas as pd
import numpy as np
from numpy import inf
from gym import spaces
from datetime import datetime
from sklearn import preprocessing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, calmar_ratio, omega_ratio
from cachetools import LRUCache, cachedmethod
import operator

from render.BitcoinTradingGraph import BitcoinTradingGraph
from util.stationarization import log_and_difference
from util.benchmarks import buy_and_hodl, rsi_divergence, sma_crossover
from util.indicators import add_indicators


class DataProvider:
    def __init__(self, df, cur_left='BTC', cur_right='USD', initial_balance=10000, commission=0.0025, **kwargs):
        self.df = df.fillna(method='bfill').reset_index()
        self.cur_right = cur_right
        self.cur_left = cur_left
        self.stationary_df = log_and_difference(
            self.df, ['Open', 'High', 'Low', 'Close', f'Volume {cur_left}', f'Volume {cur_right}'])
        self.benchmarks = [
            {
                'label': 'Buy and HODL',
                'values': buy_and_hodl(self.df['Close'], initial_balance, commission)
            },
            {
                'label': 'RSI Divergence',
                'values': rsi_divergence(self.df['Close'], initial_balance, commission)
            },
            {
                'label': 'SMA Crossover',
                'values': sma_crossover(self.df['Close'], initial_balance, commission)
            }
        ]
        self.forecast_len = kwargs.get('forecast_len', 10)
        #self.forecast_len = kwargs.get('forecast_len', 0)
        self.confidence_interval = kwargs.get('confidence_interval', 0.95)

        self.obs_shape = (1, 5 + len(self.df.columns) -
                          2 + (self.forecast_len * 3))
        #self.obs_shape = (1, 5 + len(self.df.columns) - 2)
        self.initial_balance = initial_balance
        self.commission = commission
        self.scaler_cache = LRUCache(maxsize=16384)
        self.sarimax_cache = LRUCache(maxsize=16384)

    @cachedmethod(operator.attrgetter('scaler_cache'))
    def scaled_part(self, until_idx: int):
        scaler = preprocessing.MinMaxScaler()
        features = self.stationary_df[self.stationary_df.columns.difference([
        'index', 'Date'])]
        scaled = features[:until_idx].values
        scaled[abs(scaled) == inf] = 0
        scaler = scaler.fit(scaled.astype('float64'))
        return (scaler.scale_, scaler.min_)
        

    @cachedmethod(operator.attrgetter('sarimax_cache'))
    def sarimax_forecast(self, until_idx: int):
        past_df = self.stationary_df['Close'][:
                                              until_idx]
        forecast_model = SARIMAX(past_df.values, enforce_stationarity=False)
        model_fit = forecast_model.fit(method='bfgs', disp=False)
        forecast = model_fit.get_forecast(
            steps=self.forecast_len, alpha=(1 - self.confidence_interval))
        return (forecast.predicted_mean, forecast.conf_int().flatten())


class BitcoinTradingEnv(gym.Env):
    '''A Bitcoin trading environment for OpenAI gym'''
    metadata = {'render.modes': ['human', 'system', 'rgb_array', 'none']}
    viewer = None

    def __init__(self, provider: DataProvider, reward_func='sortino', **kwargs):
        super(BitcoinTradingEnv, self).__init__()

        self.initial_balance = provider.initial_balance
        self.commission = provider.commission
        self.reward_func = reward_func

        # Actions of the format Buy 1/4, Sell 3/4, Hold (amount ignored), etc.
        self.action_space = spaces.Discrete(12)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(
            low=0, high=1, shape=provider.obs_shape, dtype=np.float16)
        self.trades = []
        self.provider = provider

    def _next_observation(self):
        until_idx = self.current_step + self.provider.forecast_len + 1

        scale_, min_ = self.provider.scaled_part(until_idx)
        features = self.provider.stationary_df[self.provider.stationary_df.columns.difference([
        'index', 'Date'])]
        scaled = features[:until_idx].values
        scaled[abs(scaled) == inf] = 0
        scaled *= scale_
        scaled += min_
        scaled = pd.DataFrame(scaled, columns=features.columns)

        # scaled, scaler = self.provider.scaled_part(until_idx)
        obs = scaled.values[-1]

        predicted_mean, forecast_conf = self.provider.sarimax_forecast(until_idx)

        obs = np.insert(obs, len(obs), predicted_mean, axis=0)
        obs = np.insert(obs, len(obs), forecast_conf, axis=0)


        scaler = preprocessing.MinMaxScaler()
        scaled_history = scaler.fit_transform(
            self.account_history.astype('float64'))

        obs = np.insert(obs, len(obs), scaled_history[:, -1], axis=0)
        obs = np.reshape(obs.astype('float16'), self.provider.obs_shape)

        return obs

    def _current_price(self):
        return self.provider.df['Close'].values[self.current_step + self.provider.forecast_len] + 0.01

    def _take_action(self, action):
        current_price = self._current_price()
        action_type = int(action / 4)
        amount = 1 / (action % 4 + 1)

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if action_type == 0:
            price = current_price * (1 + self.commission)
            btc_bought = min(self.balance * amount /
                             price, self.balance / price)
            cost = btc_bought * price

            self.btc_held += btc_bought
            self.balance -= cost
        elif action_type == 1:
            price = current_price * (1 - self.commission)
            btc_sold = self.btc_held * amount
            sales = btc_sold * price

            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({'step': self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': sales if btc_sold > 0 else cost,
                                'type': 'sell' if btc_sold > 0 else 'buy'})

        self.net_worths.append(
            self.balance + self.btc_held * current_price)

        self.account_history = np.append(self.account_history, [
            [self.balance],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

    def _reward(self):
        length = min(self.current_step, self.provider.forecast_len)
        returns = np.diff(self.net_worths)[-length:]

        if np.count_nonzero(returns) < 1:
            return 0

        if self.reward_func == 'sortino':
            reward = sortino_ratio(
                returns, annualization=365*24)
        elif self.reward_func == 'calmar':
            reward = calmar_ratio(
                returns, annualization=365*24)
        elif self.reward_func == 'omega':
            reward = omega_ratio(
                returns, annualization=365*24)
        else:
            reward = returns[-1]

        return reward if abs(reward) != inf and not np.isnan(reward) else 0

    def _done(self):
        return self.net_worths[-1] < self.initial_balance / 10 or self.current_step == len(self.provider.df) - self.provider.forecast_len - 1

    def reset(self):
        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.btc_held = 0
        self.current_step = 0

        self.account_history = np.array([
            [self.balance],
            [0],
            [0],
            [0],
            [0]
        ])
        self.trades = []

        return self._next_observation()

    def step(self, action):
        #devenv
        self._take_action(action)
        self.current_step += 1
        obs = self._next_observation()
        reward = self._reward()
        done = self._done()

        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'system':
            print('Price: ' + str(self._current_price()))
            print(
                'Bought: ' + str(self.account_history[2][self.current_step]))
            print(
                'Sold: ' + str(self.account_history[4][self.current_step]))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = BitcoinTradingGraph(self.provider.df, self.provider.cur_left, self.provider.cur_right)

            self.viewer.render(self.current_step,
                               self.net_worths, self.provider.benchmarks, self.trades)

        elif mode == 'rgb_array':
            if self.viewer is None:
                self.viewer = BitcoinTradingGraph(self.provider.df, self.provider.cur_left, self.provider.cur_right)

            self.viewer.render(self.current_step,
                               self.net_worths, self.provider.benchmarks, self.trades)
            self.viewer.fig.canvas.draw()
            buf = self.viewer.fig.canvas.tostring_rgb()
            ncols, nrows = self.viewer.fig.canvas.get_width_height()
            return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
