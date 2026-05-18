import pandas as pd
import yaml
import os
os.chdir('..')
from action_space import HOLD_ACTION, BUY_HOLD_ACTION
from trading_environment import TradingEnvironment

with open('hyperparameters.yml', 'r') as file:
    all_hyperparameter_sets = yaml.safe_load(file)
    hyperparameters = all_hyperparameter_sets["midas-hold"]
env_make_params = hyperparameters.get('env_make_params', {})
env_id = hyperparameters.get('env_id')
env = TradingEnvironment(env_make_params)
equity = 0
for i in range(50):
    for i in range(env.num_trades):
        env._process_action(BUY_HOLD_ACTION)

    _, _, terminated, _, _ = env.step(HOLD_ACTION.index)
    reward = 0
    while not terminated:
        _, equity, terminated, _, _ = env.step(HOLD_ACTION.index)

    env.reset()
    if equity > 0: env.equity_curve.append(equity)
    ep_stats = env.get_episode_stats()
    equity = 0
    os.makedirs(f'results/backtest/{env_id}', exist_ok=True)
    df = pd.DataFrame(ep_stats)
    df['id'] = env_id
    df.to_csv(f'results/backtest/{env_id}/{env_id}.csv', index=False)
