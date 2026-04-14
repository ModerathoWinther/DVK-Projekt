import os

from action_space import ACTION_SPACE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.makedirs('results', exist_ok=True)

import argparse
import itertools
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn

from dqn import DQN
from experience_replay import ReplayMemory
from trading_environment import TradingEnvironment



DATE_FORMAT = "%y-%m-%d %H:%M:%S"
RESULTS_DIR = 'results'
DEVICE = 'cpu'

class TradingAgent:

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_make_params = hyperparameters.get('env_make_params', {})
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']  #gamma
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.indicator_columns = 0
        self.parameters = hyperparameters['env_make_params']



        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RESULTS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RESULTS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RESULTS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True):

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')


        env = TradingEnvironment(params=self.parameters)
        num_actions = env.action_space.n

        num_states = env.observation_space.shape[0]

        rewards_per_episode = []
        sharpe_per_episode = []

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(DEVICE)

        if is_training:
            epsilon = self.epsilon_init

            memory = ReplayMemory(self.replay_memory_size)

            # Create the target network and make it identical to the policy network
            # with its own unique pointer.
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(DEVICE)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Adam optimizer for policy optimization
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            # Trackers for change in epsilon & N.o. steps
            epsilon_tracker = []
            step_count = 0
            best_reward = best_sharpe = - float('inf')

        # If NOT training, load eval model.
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # Main training loop, runs until dataset is complete
        for episode in itertools.count():

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)

            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:

                if is_training and random.random() < epsilon:
                    action = random.randrange(3)
                    action = torch.tensor(action, dtype=torch.int64, device=DEVICE)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                new_state, reward, terminated, truncated, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=DEVICE)
                reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            episode_sharpe = env.calculate_sharpe_ratio()
            sharpe_per_episode.append(episode_sharpe)

            # Save model when new best reward is obtained.
            if is_training:


                if episode_sharpe > best_sharpe:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}  |  New best Sharpe {episode_sharpe: 3f} (episode reward: {episode_reward:.1f}) at episode {episode}"

                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    best_sharpe = episode_sharpe

                if episode_reward > best_reward:
                    log_message = (
                        f"{datetime.now().strftime(DATE_FORMAT)} | Episode {episode} | "
                        f"Reward: {episode_reward:.2f} | "
                        f"Equity diff: {env.current_equity - env.initial_capital:.2f} | "
                        f"Sharpe: {episode_sharpe:.3f} | "
                        f"Epsilon: {epsilon:.3f}"
                    )
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_tracker, sharpe_per_episode)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_tracker.append(epsilon)


                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history, sharpe_per_episode):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        mean_sharpe = np.zeros(len(sharpe_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        for x in range(len(mean_sharpe)):
            mean_sharpe[x] = np.mean(sharpe_per_episode[max(0, x -99):(x + 1)])

        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)





        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)  # action indices, not Action objects
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float()

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()



    midas = TradingAgent(hyperparameter_set=args.hyperparameters)

    if args.train:
        midas.run(is_training=True)
    else:
        midas.run(is_training=False)