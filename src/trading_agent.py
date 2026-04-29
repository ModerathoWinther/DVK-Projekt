import os

from action_space import ACTION_SPACE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.chdir('..')
import argparse
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
        self.min_buffer_fill = hyperparameters.get('min_buffer_fill', self.mini_batch_size)
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.max_episodes = hyperparameters['max_episodes']
        self.parameters = hyperparameters['env_make_params']
        self.split = self.parameters['split']
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        print(self.split)
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.val_params = hyperparameters['validation']
        print(self.val_params)

        self.deterministic = self.val_params['deterministic']
        self.val_epsilon = self.val_params['epsilon']
        self.val_episodes = self.val_params['val_episodes']
        self.val_frequency_episode = self.val_params['val_frequency_episode']

        self.LOG_FILE = os.path.join(RESULTS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RESULTS_DIR, f'{self.env_id}.pt')
        self.GRAPH_FILE = os.path.join(RESULTS_DIR, f'{self.hyperparameter_set}.png')

    def _run_validation(self, policy_dqn: DQN, val_env: TradingEnvironment) -> tuple[float, float, float]:

        policy_dqn.eval()

        episode_rewards = []
        episode_sharpes = []
        episode_winrates = []

        for _ in range(self.val_episodes):
            state, _ = val_env.reset()
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if random.random() < self.val_epsilon:
                    action = random.randrange(len(ACTION_SPACE))
                    action = torch.tensor(action, dtype=torch.int64, device=DEVICE)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

                new_state, reward, terminated, _, _ = val_env.step(action.item())
                episode_reward += reward
                state = torch.tensor(new_state, dtype=torch.float, device=DEVICE)

            stats = val_env._calc_episode_stats()
            episode_rewards.append(episode_reward)
            episode_sharpes.append(stats['sharpe_ratio'])
            episode_winrates.append(stats['win_rate'])

        policy_dqn.train()  # restore training mode

        return (
            float(np.mean(episode_rewards)),
            float(np.mean(episode_sharpes)),
            float(np.mean(episode_winrates)),
        )

    def run(self, is_training=True):
        log_message = ""

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')


        env = TradingEnvironment(params=self.parameters)
        num_actions = env.action_space.n

        num_states = env.observation_space.shape[0]
        val_params = {**self.parameters, 'split': 'val'}
        if is_training:
            val_env = TradingEnvironment(params=val_params)
        else: val_env = None


        train_rewards = []
        train_sharpe = []
        train_winrates = []
        val_rewards = []
        val_sharpe = []
        val_winrates = []
        val_episodes_x = []
        epsilon_tracker = []

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
            best_reward = best_sharpe = best_val_sharpe =  - float('inf')

        # If NOT training, load eval model.
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # Main training loop, runs until dataset is complete
        for episode in range(self.max_episodes):
            is_val_episode = is_training and (episode % self.val_frequency_episode == 0)

            # Validation episodes run
            if is_val_episode and is_training:
                val_mean_reward, val_mean_sharpe, val_mean_winrate = self._run_validation(
                    policy_dqn, val_env
                )
                val_rewards.append(val_mean_reward)
                val_sharpe.append(val_mean_sharpe)
                val_winrates.append(val_mean_winrate)
                val_episodes_x.append(episode)

                val_log = (
                    f"[VAL] | {datetime.now().strftime(DATE_FORMAT)} | "
                    f"Episode {episode} | "
                    f"mean_reward={val_mean_reward:.2f} | "
                    f"mean_sharpe={val_mean_sharpe:.3f} | "
                    f"mean_winrate={val_mean_winrate:.2f}"
                )
                print(val_log)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(val_log + '\n')

                # Save model on best validation Sharpe — not training Sharpe
                if val_mean_sharpe > best_val_sharpe:
                    best_val_sharpe = val_mean_sharpe
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(f"[SAVED] New best val Sharpe {best_val_sharpe:.3f}\n")

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=DEVICE)

            terminated = False
            episode_reward = 0.0

            is_val_episode = (episode % self.val_frequency_episode == 0)

            while not terminated:

                if is_training and random.random() < epsilon:
                    action = random.randrange(len(ACTION_SPACE))
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

                    # Decay epsilon after each step if memory is large enough.
                    if len(memory) > self.min_buffer_fill:
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                state = new_state

            # Keep track of the rewards collected per step.
            train_rewards.append(episode_reward)
            stats = env._calc_episode_stats()
            win_rate = stats['win_rate']
            train_winrates.append(win_rate)
            episode_sharpe = stats['sharpe_ratio']
            train_sharpe.append(episode_sharpe)

            # Save model when new best reward is obtained.
            if is_training:
                epsilon_tracker.append(epsilon)
                log_message = f"[STATUS] |  {datetime.now().strftime(DATE_FORMAT)}  |  End of episode {episode}  |  n steps: {step_count} from row {env.current_step - step_count} in dataset\t\t|  win_rate: {win_rate:.2f}  |  Epsilon: {epsilon:.3f}  |  Sharpe: {episode_sharpe:.3f}\t|  (episode reward: {episode_reward:.1f})"
                if episode_sharpe > best_sharpe:
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    best_sharpe = episode_sharpe

                if episode_reward > best_reward:
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(
                        train_rewards, train_sharpe, train_winrates,
                        val_rewards, val_sharpe, val_winrates,
                        val_episodes_x, epsilon_tracker
                    )
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0
                print(log_message)
            elif self.split == 'test':
                print("reward", episode_reward)
                print(env.get_episode_stats())

    def save_graph(self,
                   train_rewards, train_sharpes, train_winrates,
                   val_rewards, val_sharpes, val_winrates,
                   val_x,
                   epsilon_history):

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        def moving_avg(data, window=20):
            return np.array([np.mean(data[max(0, i - window):i + 1]) for i in range(len(data))])

        train_x = list(range(len(train_rewards)))

        # Reward
        axes[0, 0].plot(train_x, moving_avg(train_rewards), color='blue', label='Train')
        if val_rewards:
            axes[0, 0].plot(val_x, val_rewards, color='cyan', linestyle='--', marker='o', label='Val')
        axes[0, 0].set_title('Episode Reward (20-ep MA)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].legend()

        # Sharpe
        axes[0, 1].plot(train_x, moving_avg(train_sharpes), color='green', label='Train')
        if val_sharpes:
            axes[0, 1].plot(val_x, val_sharpes, color='lime', linestyle='--', marker='o', label='Val')
        axes[0, 1].set_title('Sharpe Ratio (20-ep MA)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].legend()

        # Win rate
        axes[0, 2].plot(train_x, moving_avg(train_winrates), color='orange', label='Train')
        if val_winrates:
            axes[0, 2].plot(val_x, val_winrates, color='gold', linestyle='--', marker='o', label='Val')
        axes[0, 2].axhline(y=0.333, color='black', linestyle=':', label='Break-even')
        axes[0, 2].set_title('Win Rate (20-ep MA)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].legend()

        # Epsilon
        axes[1, 0].plot(epsilon_history, color='red')
        axes[0, 2].axhline(y=0.05, color='black', linestyle=':', label='Epsilon min')
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episodes')

        # Train vs Val Sharpe scatter — shows overfitting directly
        if val_sharpes and len(val_sharpes) > 1:
            axes[1, 1].plot(val_x, train_sharpes[:len(val_x)],
                            color='green', alpha=0.5, label='Train Sharpe at val point')
            axes[1, 1].plot(val_x, val_sharpes,
                            color='lime', linestyle='--', label='Val Sharpe')
            axes[1, 1].fill_between(val_x,
                                    train_sharpes[:len(val_x)],
                                    val_sharpes,
                                    alpha=0.2, color='red', label='Overfitting gap')
            axes[1, 1].set_title('Train vs Val Sharpe (overfitting monitor)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].legend()

        # Val Sharpe over time — the primary learning signal
        if val_sharpes:
            axes[1, 2].plot(val_x, val_sharpes, color='purple', marker='o')
            axes[1, 2].axhline(y=0, color='black', linestyle=':')
            axes[1, 2].set_title('Validation Sharpe Progress')
            axes[1, 2].set_xlabel('Training Episode')

        plt.tight_layout()
        pdf_path = self.GRAPH_FILE.replace('.png', '.pdf')
        fig.savefig(pdf_path, format='pdf')
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float()

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
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