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

num_episodes = hyperparameters['max_episodes']  # or however many you want to run
counter = 1

for counter in range(num_episodes):
    for i in range(env.num_trades):
        env._process_action(BUY_HOLD_ACTION)

    _, _, terminated, _, _ = env.step(HOLD_ACTION.index)
    while not terminated:
        _, _, terminated, _, _ = env.step(HOLD_ACTION.index)

    env.reset()
    counter += 1

# Save to CSV
os.makedirs(f'results/backtest/{env_id}', exist_ok=True)
df = pd.DataFrame(env.get_episode_stats())
df['id'] = env_id
df.to_csv(f'results/backtest/{env_id}/{env_id}.csv', index=False)