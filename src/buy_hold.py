import yaml

from action_space import HOLD_ACTION, BUY_HOLD_ACTION
from trading_environment import TradingEnvironment

with open('../hyperparameters.yml', 'r') as file:
    all_hyperparameter_sets = yaml.safe_load(file)
    hyperparameters = all_hyperparameter_sets["midas-train1"]
env_make_params = hyperparameters.get('env_make_params', {})

env = TradingEnvironment(env_make_params)

for i in range(env.num_trades):
    env._process_action(BUY_HOLD_ACTION)

print(env.trades_state)

_, _, terminated, _, _ = env.step(HOLD_ACTION.index)
print(env.trades_state)
reward = 0
while not terminated:
    _, _, terminated, _, _ = env.step(HOLD_ACTION.index)

env.reset()
print(env.get_episode_stats())