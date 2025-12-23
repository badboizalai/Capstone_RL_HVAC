"""
Critic Network for DDPG (Twin Q-Networks)
"""
import torch
import torch.nn as nn


class Critic(nn.Module):
    """Twin Q-Network Critic: (state, action) -> Q-value"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4_q1 = nn.Linear(hidden_dim // 2, 1)
        
        # Q2 network
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4_q2 = nn.Linear(hidden_dim // 2, 1)
        
        self.ln1_q1 = nn.LayerNorm(hidden_dim)
        self.ln2_q1 = nn.LayerNorm(hidden_dim)
        self.ln3_q1 = nn.LayerNorm(hidden_dim // 2)
        
        self.ln1_q2 = nn.LayerNorm(hidden_dim)
        self.ln2_q2 = nn.LayerNorm(hidden_dim)
        self.ln3_q2 = nn.LayerNorm(hidden_dim // 2)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for layer in [self.fc1_q1, self.fc2_q1, self.fc3_q1,
                      self.fc1_q2, self.fc2_q2, self.fc3_q2]:
            nn.init.xavier_uniform_(layer.weight)
        nn.init.uniform_(self.fc4_q1.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4_q2.weight, -3e-3, 3e-3)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = torch.relu(self.ln1_q1(self.fc1_q1(x)))
        q1 = torch.relu(self.ln2_q1(self.fc2_q1(q1)))
        q1 = torch.relu(self.ln3_q1(self.fc3_q1(q1)))
        q1 = self.fc4_q1(q1)
        
        # Q2
        q2 = torch.relu(self.ln1_q2(self.fc1_q2(x)))
        q2 = torch.relu(self.ln2_q2(self.fc2_q2(q2)))
        q2 = torch.relu(self.ln3_q2(self.fc3_q2(q2)))
        q2 = self.fc4_q2(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """Return only Q1 for actor loss"""
        x = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.ln1_q1(self.fc1_q1(x)))
        q1 = torch.relu(self.ln2_q1(self.fc2_q1(q1)))
        q1 = torch.relu(self.ln3_q1(self.fc3_q1(q1)))
        q1 = self.fc4_q1(q1)
        return q1
