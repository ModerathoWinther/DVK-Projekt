import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, enable_dueling_dqn):
        super(DQN, self).__init__()

        self.enable_dueling_dqn=enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            self.fc_value = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, 1)
            self.fc_advantages = nn.Linear(hidden_dim, hidden_dim)
            self.advantages = nn.Linear(hidden_dim, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            Q = self.output(x)

        return Q
