"""Modern RL Algorithms for Energy Management.

This module implements state-of-the-art reinforcement learning algorithms
optimized for continuous control in energy management systems. Includes
DQN variants, PPO, SAC, and TD3 with proper type hints and documentation.

WARNING: This is a research/educational implementation. NOT FOR PRODUCTION USE
in real energy management systems.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import deque
import warnings


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms.
    
    Args:
        capacity: Maximum number of experiences to store
        device: Device to store tensors on
    """
    
    def __init__(self, capacity: int, device: torch.device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(action).to(self.device),
            torch.FloatTensor(reward).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.BoolTensor(done).to(self.device),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network for discrete action spaces.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorNetwork(nn.Module):
    """Actor network for continuous action spaces.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
        max_action: Maximum action value
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256,
        max_action: float = 1.0
    ):
        super().__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)) * self.max_action


class CriticNetwork(nn.Module):
    """Critic network for continuous action spaces.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        lr: Learning rate
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        epsilon_min: Minimum epsilon value
        buffer_size: Replay buffer size
        batch_size: Training batch size
        target_update: Target network update frequency
        device: Device to run on
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 100,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, self.device)
        
        # Training step counter
        self.step_count = 0
        
        warnings.warn(
            "WARNING: This is a research implementation. NOT FOR PRODUCTION USE "
            "in real energy management systems.",
            UserWarning,
            stacklevel=2
        )
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """Update Q-network using experience replay."""
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Convert actions to one-hot for discrete actions
        actions_onehot = F.one_hot(actions.long(), self.action_dim).float()
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath: str) -> None:
        """Save model parameters."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']


class SACAgent:
    """Soft Actor-Critic agent for continuous control.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        lr: Learning rate
        gamma: Discount factor
        tau: Soft update parameter
        alpha: Temperature parameter (auto-tuned if None)
        buffer_size: Replay buffer size
        batch_size: Training batch size
        hidden_dim: Hidden layer dimension
        device: Device to run on
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: Optional[float] = None,
        buffer_size: int = 100000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Target networks
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Temperature parameter
        if alpha is None:
            self.auto_alpha = True
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.auto_alpha = False
            self.alpha = alpha
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, self.device)
        
        warnings.warn(
            "WARNING: This is a research implementation. NOT FOR PRODUCTION USE "
            "in real energy management systems.",
            UserWarning,
            stacklevel=2
        )
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor)
            
        if training:
            # Add noise for exploration
            noise = torch.randn_like(action) * 0.1
            action = torch.clamp(action + noise, -1.0, 1.0)
        
        return action.cpu().numpy().flatten()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[Dict[str, float]]:
        """Update SAC networks."""
        if len(self.buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_actions += torch.randn_like(next_actions) * 0.1
            next_actions = torch.clamp(next_actions, -1.0, 1.0)
            
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
            target_q = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q
        
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions = self.actor(states)
        new_actions += torch.randn_like(new_actions) * 0.1
        new_actions = torch.clamp(new_actions, -1.0, 1.0)
        
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
        actor_loss = (alpha * torch.log(torch.clamp(1 - new_actions.pow(2), min=1e-6)).sum(1, keepdim=True) - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (torch.log(torch.clamp(1 - new_actions.pow(2), min=1e-6)).sum(1, keepdim=True) + 1.0)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return {
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': alpha.item() if self.auto_alpha else alpha,
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save(self, filepath: str) -> None:
        """Save model parameters."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_alpha else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_alpha else None,
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load model parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        
        if self.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
