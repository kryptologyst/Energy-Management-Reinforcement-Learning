"""Test suite for Energy Management RL project.

This module contains comprehensive tests for all components of the
energy management reinforcement learning system.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from src.environments.energy_management import EnergyManagementEnv
from src.algorithms.modern_rl import DQNAgent, SACAgent, ReplayBuffer
from src.training.trainer import EnergyRLTrainer, EnergyRLEvaluator
from src.utils.config import get_default_config, load_config


class TestEnergyManagementEnv:
    """Test cases for EnergyManagementEnv."""
    
    def test_env_initialization(self):
        """Test environment initialization."""
        env = EnergyManagementEnv()
        
        assert env.observation_space.shape == (5,)
        assert env.action_space.shape == (4,)
        assert env.max_storage_capacity == 100.0
        assert env.max_demand == 50.0
        assert env.max_generation == 80.0
    
    def test_env_reset(self):
        """Test environment reset."""
        env = EnergyManagementEnv(seed=42)
        obs, info = env.reset()
        
        assert obs.shape == (5,)
        assert isinstance(info, dict)
        assert 'step' in info
        assert 'storage_level' in info
    
    def test_env_step(self):
        """Test environment step."""
        env = EnergyManagementEnv(seed=42)
        obs, _ = env.reset()
        
        action = np.array([0.5, 0.3, 0.2, 0.1])
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert next_obs.shape == (5,)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_env_constraints(self):
        """Test environment constraint handling."""
        env = EnergyManagementEnv(seed=42)
        obs, _ = env.reset()
        
        # Test extreme actions
        extreme_action = np.array([1.0, 1.0, 1.0, 1.0])
        next_obs, reward, terminated, truncated, info = env.step(extreme_action)
        
        # Should not crash and should handle constraints
        assert not np.isnan(reward)
        assert not np.isinf(reward)
    
    def test_env_render(self):
        """Test environment rendering."""
        env = EnergyManagementEnv(seed=42)
        obs, _ = env.reset()
        
        # Test human mode
        env.render(mode="human")
        
        # Test rgb_array mode
        rgb_array = env.render(mode="rgb_array")
        assert rgb_array is not None
        assert rgb_array.shape[2] == 3  # RGB channels


class TestReplayBuffer:
    """Test cases for ReplayBuffer."""
    
    def test_buffer_initialization(self):
        """Test replay buffer initialization."""
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=1000, device=device)
        
        assert buffer.buffer.maxlen == 1000
        assert buffer.device == device
    
    def test_buffer_push_sample(self):
        """Test buffer push and sample operations."""
        device = torch.device("cpu")
        buffer = ReplayBuffer(capacity=1000, device=device)
        
        # Push some transitions
        for i in range(10):
            state = np.random.rand(5)
            action = np.random.rand(4)
            reward = np.random.rand()
            next_state = np.random.rand(5)
            done = i % 3 == 0
            
            buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 10
        
        # Sample batch
        batch = buffer.sample(5)
        assert len(batch) == 5
        assert all(isinstance(tensor, torch.Tensor) for tensor in batch)


class TestDQNAgent:
    """Test cases for DQNAgent."""
    
    def test_agent_initialization(self):
        """Test DQN agent initialization."""
        device = torch.device("cpu")
        agent = DQNAgent(
            state_dim=5,
            action_dim=4,
            device=device
        )
        
        assert agent.state_dim == 5
        assert agent.action_dim == 4
        assert agent.device == device
        assert agent.epsilon == 1.0
    
    def test_action_selection(self):
        """Test action selection."""
        device = torch.device("cpu")
        agent = DQNAgent(state_dim=5, action_dim=4, device=device)
        
        state = np.random.rand(5)
        action = agent.select_action(state, training=True)
        
        assert isinstance(action, int)
        assert 0 <= action < 4
    
    def test_agent_update(self):
        """Test agent update."""
        device = torch.device("cpu")
        agent = DQNAgent(state_dim=5, action_dim=4, device=device)
        
        # Add some transitions to buffer
        for _ in range(100):
            state = np.random.rand(5)
            action = np.random.randint(4)
            reward = np.random.rand()
            next_state = np.random.rand(5)
            done = np.random.choice([True, False])
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Test update
        loss = agent.update()
        assert loss is not None
        assert isinstance(loss, float)


class TestSACAgent:
    """Test cases for SACAgent."""
    
    def test_agent_initialization(self):
        """Test SAC agent initialization."""
        device = torch.device("cpu")
        agent = SACAgent(
            state_dim=5,
            action_dim=4,
            device=device
        )
        
        assert agent.state_dim == 5
        assert agent.action_dim == 4
        assert agent.device == device
    
    def test_action_selection(self):
        """Test action selection."""
        device = torch.device("cpu")
        agent = SACAgent(state_dim=5, action_dim=4, device=device)
        
        state = np.random.rand(5)
        action = agent.select_action(state, training=True)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (4,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
    
    def test_agent_update(self):
        """Test agent update."""
        device = torch.device("cpu")
        agent = SACAgent(state_dim=5, action_dim=4, device=device)
        
        # Add some transitions to buffer
        for _ in range(256):
            state = np.random.rand(5)
            action = np.random.rand(4)
            reward = np.random.rand()
            next_state = np.random.rand(5)
            done = np.random.choice([True, False])
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Test update
        losses = agent.update()
        assert losses is not None
        assert isinstance(losses, dict)


class TestTrainer:
    """Test cases for EnergyRLTrainer."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        env = EnergyManagementEnv()
        agent = SACAgent(state_dim=5, action_dim=4)
        config = get_default_config()
        
        trainer = EnergyRLTrainer(env, agent, config)
        
        assert trainer.env == env
        assert trainer.agent == agent
        assert trainer.config == config
    
    def test_train_episode(self):
        """Test single episode training."""
        env = EnergyManagementEnv()
        agent = SACAgent(state_dim=5, action_dim=4)
        config = get_default_config()
        
        trainer = EnergyRLTrainer(env, agent, config)
        
        episode_reward, episode_length, losses = trainer._train_episode(max_steps=10)
        
        assert isinstance(episode_reward, float)
        assert isinstance(episode_length, int)
        assert episode_length <= 10
        assert isinstance(losses, list)
    
    def test_evaluation(self):
        """Test agent evaluation."""
        env = EnergyManagementEnv()
        agent = SACAgent(state_dim=5, action_dim=4)
        config = get_default_config()
        
        trainer = EnergyRLTrainer(env, agent, config)
        
        eval_reward = trainer.evaluate(num_episodes=3, render=False)
        
        assert isinstance(eval_reward, float)


class TestEvaluator:
    """Test cases for EnergyRLEvaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        env = EnergyManagementEnv()
        evaluator = EnergyRLEvaluator(env)
        
        assert evaluator.env == env
    
    def test_agent_evaluation(self):
        """Test agent evaluation."""
        env = EnergyManagementEnv()
        agent = SACAgent(state_dim=5, action_dim=4)
        evaluator = EnergyRLEvaluator(env)
        
        metrics = evaluator.evaluate_agent(agent, num_episodes=5)
        
        assert isinstance(metrics, dict)
        assert 'rewards' in metrics
        assert 'success_rate' in metrics
        assert 'constraint_violations' in metrics


class TestConfig:
    """Test cases for configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = get_default_config()
        
        assert 'env' in config
        assert 'agent' in config
        assert 'training' in config
        assert 'evaluation' in config
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = load_config()
        
        assert config is not None
        assert hasattr(config, 'env')
        assert hasattr(config, 'agent')


if __name__ == "__main__":
    pytest.main([__file__])
