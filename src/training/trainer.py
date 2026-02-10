"""Training and Evaluation Module for Energy Management RL.

This module provides comprehensive training and evaluation pipelines for
energy management reinforcement learning agents. Includes proper logging,
checkpointing, and statistical evaluation.

WARNING: This is a research/educational implementation. NOT FOR PRODUCTION USE
in real energy management systems.
"""

from __future__ import annotations

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
from pathlib import Path
import time
from tqdm import tqdm
import warnings
from scipy import stats
import logging

from ..environments.energy_management import EnergyManagementEnv
from ..algorithms.modern_rl import DQNAgent, SACAgent


class EnergyRLTrainer:
    """Training pipeline for energy management RL agents.
    
    Args:
        env: Energy management environment
        agent: RL agent to train
        config: Training configuration dictionary
        log_dir: Directory for logging and checkpoints
        device: Device to run training on
    """
    
    def __init__(
        self,
        env: EnergyManagementEnv,
        agent: Union[DQNAgent, SACAgent],
        config: Dict[str, Any],
        log_dir: str = "logs",
        device: Optional[torch.device] = None,
    ):
        self.env = env
        self.agent = agent
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.training_losses: List[float] = []
        self.eval_rewards: List[float] = []
        
        # Setup logging
        self._setup_logging()
        
        warnings.warn(
            "WARNING: This is a research implementation. NOT FOR PRODUCTION USE "
            "in real energy management systems.",
            UserWarning,
            stacklevel=2
        )
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train(
        self, 
        num_episodes: int,
        eval_frequency: int = 100,
        save_frequency: int = 500,
        max_steps_per_episode: int = 1000,
    ) -> Dict[str, List[float]]:
        """Train the RL agent.
        
        Args:
            num_episodes: Number of training episodes
            eval_frequency: Frequency of evaluation episodes
            save_frequency: Frequency of model saving
            max_steps_per_episode: Maximum steps per episode
            
        Returns:
            Dictionary containing training metrics
        """
        self.logger.info(f"Starting training for {num_episodes} episodes")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Agent type: {type(self.agent).__name__}")
        
        start_time = time.time()
        
        for episode in tqdm(range(num_episodes), desc="Training"):
            episode_reward, episode_length, losses = self._train_episode(max_steps_per_episode)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if losses:
                self.training_losses.extend(losses)
            
            # Evaluation
            if episode % eval_frequency == 0:
                eval_reward = self.evaluate(num_episodes=10, render=False)
                self.eval_rewards.append(eval_reward)
                self.logger.info(
                    f"Episode {episode}: Reward={episode_reward:.2f}, "
                    f"Length={episode_length}, Eval Reward={eval_reward:.2f}"
                )
            
            # Save checkpoint
            if episode % save_frequency == 0:
                self._save_checkpoint(episode)
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model and training metrics
        self._save_final_model()
        self._save_training_metrics()
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses,
            'eval_rewards': self.eval_rewards,
        }
    
    def _train_episode(self, max_steps: int) -> Tuple[float, int, List[float]]:
        """Train for one episode."""
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        losses = []
        
        for step in range(max_steps):
            # Select action
            if isinstance(self.agent, DQNAgent):
                action = self.agent.select_action(state, training=True)
                action_array = np.array([action])  # Convert to array for consistency
            else:
                action_array = self.agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action_array)
            done = terminated or truncated
            
            # Store transition
            if isinstance(self.agent, DQNAgent):
                self.agent.store_transition(state, action, reward, next_state, done)
            else:
                self.agent.store_transition(state, action_array, reward, next_state, done)
            
            # Update agent
            loss = self.agent.update()
            if loss is not None:
                if isinstance(loss, dict):
                    losses.append(sum(loss.values()) / len(loss))
                else:
                    losses.append(loss)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        return episode_reward, episode_length, losses
    
    def evaluate(
        self, 
        num_episodes: int = 10, 
        render: bool = False,
        deterministic: bool = True
    ) -> float:
        """Evaluate the trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Average evaluation reward
        """
        eval_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            
            while True:
                # Select action (no exploration during evaluation)
                if isinstance(self.agent, DQNAgent):
                    action = self.agent.select_action(state, training=False)
                    action_array = np.array([action])
                else:
                    action_array = self.agent.select_action(state, training=False)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action_array)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                
                if render:
                    self.env.render()
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
        
        avg_reward = np.mean(eval_rewards)
        return avg_reward
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.log_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_episode_{episode}.pth"
        self.agent.save(str(checkpoint_path))
        
        # Save training state
        training_state = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'eval_rewards': self.eval_rewards,
            'config': self.config,
        }
        
        state_path = checkpoint_dir / f"training_state_{episode}.json"
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
    
    def _save_final_model(self) -> None:
        """Save final trained model."""
        model_path = self.log_dir / "final_model.pth"
        self.agent.save(str(model_path))
        self.logger.info(f"Final model saved to {model_path}")
    
    def _save_training_metrics(self) -> None:
        """Save training metrics to files."""
        # Save raw metrics
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses,
            'eval_rewards': self.eval_rewards,
        }
        
        metrics_path = self.log_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create plots
        self._create_training_plots()
    
    def _create_training_plots(self) -> None:
        """Create training visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Training losses
        if self.training_losses:
            axes[1, 0].plot(self.training_losses)
            axes[1, 0].set_title('Training Losses')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Evaluation rewards
        if self.eval_rewards:
            eval_episodes = np.arange(0, len(self.eval_rewards)) * 100  # Assuming eval every 100 episodes
            axes[1, 1].plot(eval_episodes, self.eval_rewards)
            axes[1, 1].set_title('Evaluation Rewards')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.log_dir / "training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plots saved to {plot_path}")


class EnergyRLEvaluator:
    """Comprehensive evaluation suite for energy management RL agents.
    
    Args:
        env: Energy management environment
        log_dir: Directory for evaluation results
    """
    
    def __init__(self, env: EnergyManagementEnv, log_dir: str = "evaluation"):
        self.env = env
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        warnings.warn(
            "WARNING: This is a research implementation. NOT FOR PRODUCTION USE "
            "in real energy management systems.",
            UserWarning,
            stacklevel=2
        )
    
    def evaluate_agent(
        self,
        agent: Union[DQNAgent, SACAgent],
        num_episodes: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Comprehensive evaluation of an agent.
        
        Args:
            agent: Trained RL agent
            num_episodes: Number of evaluation episodes
            seeds: List of random seeds for reproducibility
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if seeds is None:
            seeds = list(range(num_episodes))
        
        all_rewards = []
        all_lengths = []
        all_costs = []
        all_storage_levels = []
        all_constraint_violations = []
        
        for i, seed in enumerate(seeds):
            state, _ = self.env.reset(seed=seed)
            episode_reward = 0.0
            episode_length = 0
            episode_costs = []
            episode_storage = []
            constraint_violations = 0
            
            while True:
                # Select action
                if isinstance(agent, DQNAgent):
                    action = agent.select_action(state, training=False)
                    action_array = np.array([action])
                else:
                    action_array = agent.select_action(state, training=False)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action_array)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                episode_costs.append(info.get('energy_cost', 0))
                episode_storage.append(info.get('storage_level', 0))
                
                # Check constraint violations
                if info.get('storage_level', 0) < 0 or info.get('storage_level', 0) > self.env.max_storage_capacity:
                    constraint_violations += 1
                
                state = next_state
                
                if done:
                    break
            
            all_rewards.append(episode_reward)
            all_lengths.append(episode_length)
            all_costs.append(sum(episode_costs))
            all_storage_levels.append(np.mean(episode_storage))
            all_constraint_violations.append(constraint_violations)
        
        # Calculate statistics
        metrics = self._calculate_statistics(
            all_rewards, all_lengths, all_costs, 
            all_storage_levels, all_constraint_violations
        )
        
        # Save results
        self._save_evaluation_results(metrics, agent)
        
        return metrics
    
    def _calculate_statistics(
        self,
        rewards: List[float],
        lengths: List[int],
        costs: List[float],
        storage_levels: List[float],
        constraint_violations: List[int],
    ) -> Dict[str, Any]:
        """Calculate comprehensive evaluation statistics."""
        rewards = np.array(rewards)
        lengths = np.array(lengths)
        costs = np.array(costs)
        storage_levels = np.array(storage_levels)
        constraint_violations = np.array(constraint_violations)
        
        # Basic statistics
        metrics = {
            'num_episodes': len(rewards),
            'rewards': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'median': float(np.median(rewards)),
                'q25': float(np.percentile(rewards, 25)),
                'q75': float(np.percentile(rewards, 75)),
            },
            'episode_lengths': {
                'mean': float(np.mean(lengths)),
                'std': float(np.std(lengths)),
                'min': int(np.min(lengths)),
                'max': int(np.max(lengths)),
            },
            'energy_costs': {
                'mean': float(np.mean(costs)),
                'std': float(np.std(costs)),
                'min': float(np.min(costs)),
                'max': float(np.max(costs)),
            },
            'storage_utilization': {
                'mean': float(np.mean(storage_levels)),
                'std': float(np.std(storage_levels)),
            },
            'constraint_violations': {
                'total': int(np.sum(constraint_violations)),
                'per_episode': float(np.mean(constraint_violations)),
                'episodes_with_violations': int(np.sum(constraint_violations > 0)),
            },
        }
        
        # Confidence intervals
        ci_95 = stats.t.interval(0.95, len(rewards)-1, loc=np.mean(rewards), scale=stats.sem(rewards))
        metrics['rewards']['ci_95'] = [float(ci_95[0]), float(ci_95[1])]
        
        # Success rate (episodes with positive reward)
        success_rate = np.mean(rewards > 0)
        metrics['success_rate'] = float(success_rate)
        
        # Sample efficiency (steps to reach threshold)
        threshold = np.mean(rewards) * 0.8  # 80% of mean reward
        steps_to_threshold = []
        for i, reward in enumerate(rewards):
            if reward >= threshold:
                steps_to_threshold.append(i + 1)
        
        if steps_to_threshold:
            metrics['sample_efficiency'] = {
                'steps_to_threshold': float(np.mean(steps_to_threshold)),
                'threshold': float(threshold),
            }
        
        return metrics
    
    def _save_evaluation_results(self, metrics: Dict[str, Any], agent: Union[DQNAgent, SACAgent]) -> None:
        """Save evaluation results to files."""
        # Save metrics as JSON
        results_path = self.log_dir / f"evaluation_results_{type(agent).__name__}.json"
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create evaluation plots
        self._create_evaluation_plots(metrics, agent)
    
    def _create_evaluation_plots(self, metrics: Dict[str, Any], agent: Union[DQNAgent, SACAgent]) -> None:
        """Create evaluation visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward distribution
        rewards = metrics['rewards']
        axes[0, 0].hist([rewards['mean']], bins=20, alpha=0.7, label='Mean Reward')
        axes[0, 0].axvline(rewards['mean'], color='red', linestyle='--', label=f'Mean: {rewards["mean"]:.2f}')
        axes[0, 0].axvline(rewards['ci_95'][0], color='orange', linestyle=':', label=f'95% CI: [{rewards["ci_95"][0]:.2f}, {rewards["ci_95"][1]:.2f}]')
        axes[0, 0].axvline(rewards['ci_95'][1], color='orange', linestyle=':')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode length distribution
        lengths = metrics['episode_lengths']
        axes[0, 1].bar(['Mean'], [lengths['mean']], alpha=0.7)
        axes[0, 1].set_title('Episode Length')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Energy cost analysis
        costs = metrics['energy_costs']
        axes[1, 0].bar(['Mean Cost'], [costs['mean']], alpha=0.7, color='red')
        axes[1, 0].set_title('Energy Cost')
        axes[1, 0].set_ylabel('Cost ($)')
        axes[1, 0].grid(True)
        
        # Constraint violations
        violations = metrics['constraint_violations']
        axes[1, 1].bar(['Total Violations', 'Episodes with Violations'], 
                      [violations['total'], violations['episodes_with_violations']], 
                      alpha=0.7, color='orange')
        axes[1, 1].set_title('Constraint Violations')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True)
        
        plt.suptitle(f'Evaluation Results - {type(agent).__name__}', fontsize=16)
        plt.tight_layout()
        
        plot_path = self.log_dir / f"evaluation_plots_{type(agent).__name__}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to {plot_path}")
    
    def compare_agents(
        self,
        agents: Dict[str, Union[DQNAgent, SACAgent]],
        num_episodes: int = 50,
    ) -> pd.DataFrame:
        """Compare multiple agents and create a leaderboard.
        
        Args:
            agents: Dictionary mapping agent names to agent objects
            num_episodes: Number of episodes per agent
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, agent in agents.items():
            print(f"Evaluating {name}...")
            metrics = self.evaluate_agent(agent, num_episodes)
            
            results.append({
                'Agent': name,
                'Mean Reward': metrics['rewards']['mean'],
                'Reward Std': metrics['rewards']['std'],
                'Success Rate': metrics['success_rate'],
                'Mean Cost': metrics['energy_costs']['mean'],
                'Constraint Violations': metrics['constraint_violations']['per_episode'],
                'Storage Utilization': metrics['storage_utilization']['mean'],
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('Mean Reward', ascending=False)
        
        # Save leaderboard
        leaderboard_path = self.log_dir / "agent_leaderboard.csv"
        df.to_csv(leaderboard_path, index=False)
        
        print(f"\nAgent Leaderboard:")
        print(df.to_string(index=False))
        print(f"\nLeaderboard saved to {leaderboard_path}")
        
        return df
