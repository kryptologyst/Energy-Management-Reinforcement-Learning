"""Main training script for Energy Management RL.

This script provides a complete training pipeline for energy management
reinforcement learning agents with proper configuration management,
logging, and evaluation.

WARNING: This is a research/educational implementation. NOT FOR PRODUCTION USE
in real energy management systems.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf

from src.environments.energy_management import EnergyManagementEnv
from src.algorithms.modern_rl import DQNAgent, SACAgent
from src.training.trainer import EnergyRLTrainer, EnergyRLEvaluator
from src.utils.config import load_config, get_device_config, setup_reproducibility


def create_agent(config: OmegaConf, env: EnergyManagementEnv, device: torch.device):
    """Create RL agent based on configuration.
    
    Args:
        config: Configuration object
        env: Environment instance
        device: Device to run on
        
    Returns:
        Configured RL agent
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if config.agent.type.lower() == 'dqn':
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=config.agent.lr,
            gamma=config.agent.gamma,
            epsilon=config.dqn.epsilon,
            epsilon_decay=config.dqn.epsilon_decay,
            epsilon_min=config.dqn.epsilon_min,
            buffer_size=config.agent.buffer_size,
            batch_size=config.agent.batch_size,
            target_update=config.dqn.target_update,
            device=device,
        )
    elif config.agent.type.lower() == 'sac':
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=config.agent.lr,
            gamma=config.agent.gamma,
            tau=config.sac.tau,
            alpha=config.sac.alpha,
            buffer_size=config.agent.buffer_size,
            batch_size=config.agent.batch_size,
            hidden_dim=config.agent.hidden_dim,
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {config.agent.type}")
    
    return agent


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Energy Management RL Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--num-episodes", type=int, help="Number of training episodes")
    parser.add_argument("--agent-type", type=str, choices=['dqn', 'sac'], help="Agent type")
    parser.add_argument("--log-dir", type=str, help="Logging directory")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--model-path", type=str, help="Path to trained model for evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.num_episodes:
        config.training.num_episodes = args.num_episodes
    if args.agent_type:
        config.agent.type = args.agent_type
    if args.log_dir:
        config.training.log_dir = args.log_dir
    
    # Setup reproducibility
    setup_reproducibility(config)
    
    # Get device
    device_str = get_device_config(config)
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Create environment
    env = EnergyManagementEnv(
        max_storage_capacity=config.env.max_storage_capacity,
        max_demand=config.env.max_demand,
        max_generation=config.env.max_generation,
        price_variability=config.env.price_variability,
        episode_length=config.env.episode_length,
        seed=config.env.seed,
    )
    
    print(f"Environment created: {env.observation_space.shape} -> {env.action_space.shape}")
    
    # Create agent
    agent = create_agent(config, env, device)
    print(f"Agent created: {type(agent).__name__}")
    
    if args.eval_only:
        # Evaluation only mode
        if args.model_path:
            agent.load(args.model_path)
            print(f"Loaded model from {args.model_path}")
        
        evaluator = EnergyRLEvaluator(env, config.evaluation.eval_dir)
        metrics = evaluator.evaluate_agent(agent, config.evaluation.num_episodes)
        
        print("\nEvaluation Results:")
        print(f"Mean Reward: {metrics['rewards']['mean']:.2f} ± {metrics['rewards']['std']:.2f}")
        print(f"Success Rate: {metrics['success_rate']:.2%}")
        print(f"Mean Cost: ${metrics['energy_costs']['mean']:.2f}")
        print(f"Constraint Violations: {metrics['constraint_violations']['per_episode']:.2f}")
        
    else:
        # Training mode
        trainer = EnergyRLTrainer(env, agent, config, config.training.log_dir, device)
        
        print(f"Starting training for {config.training.num_episodes} episodes...")
        training_metrics = trainer.train(
            num_episodes=config.training.num_episodes,
            eval_frequency=config.training.eval_frequency,
            save_frequency=config.training.save_frequency,
            max_steps_per_episode=config.training.max_steps_per_episode,
        )
        
        print("Training completed!")
        
        # Final evaluation
        print("Running final evaluation...")
        evaluator = EnergyRLEvaluator(env, config.evaluation.eval_dir)
        final_metrics = evaluator.evaluate_agent(agent, config.evaluation.num_episodes)
        
        print("\nFinal Evaluation Results:")
        print(f"Mean Reward: {final_metrics['rewards']['mean']:.2f} ± {final_metrics['rewards']['std']:.2f}")
        print(f"Success Rate: {final_metrics['success_rate']:.2%}")
        print(f"Mean Cost: ${final_metrics['energy_costs']['mean']:.2f}")
        print(f"Constraint Violations: {final_metrics['constraint_violations']['per_episode']:.2f}")
        
        # Save final configuration
        config_path = Path(config.training.log_dir) / "final_config.yaml"
        OmegaConf.save(config, config_path)
        print(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    main()
