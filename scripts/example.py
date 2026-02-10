#!/usr/bin/env python3
"""Quick example script for Energy Management RL.

This script demonstrates basic usage of the energy management RL system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from src.environments.energy_management import EnergyManagementEnv
from src.algorithms.modern_rl import SACAgent
from src.training.trainer import EnergyRLTrainer, EnergyRLEvaluator
from src.utils.config import load_config, get_device_config, setup_reproducibility


def main():
    """Run a quick example."""
    print("⚡ Energy Management RL - Quick Example")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    device = torch.device(get_device_config(config))
    setup_reproducibility(config)
    
    print(f"Device: {device}")
    print(f"Agent Type: {config.agent.type}")
    
    # Create environment
    env = EnergyManagementEnv(
        max_storage_capacity=100.0,
        max_demand=50.0,
        max_generation=80.0,
        seed=42
    )
    
    print(f"Environment: {env.observation_space.shape} -> {env.action_space.shape}")
    
    # Create agent
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device
    )
    
    print("Agent created successfully")
    
    # Quick training
    print("\nTraining for 50 episodes...")
    trainer = EnergyRLTrainer(env, agent, config, "example_logs", device)
    metrics = trainer.train(num_episodes=50, eval_frequency=10)
    
    print(f"Training completed!")
    print(f"Final episode reward: {metrics['episode_rewards'][-1]:.2f}")
    
    # Quick evaluation
    print("\nEvaluating agent...")
    evaluator = EnergyRLEvaluator(env, "example_eval")
    eval_metrics = evaluator.evaluate_agent(agent, num_episodes=10)
    
    print(f"Evaluation Results:")
    print(f"  Mean Reward: {eval_metrics['rewards']['mean']:.2f}")
    print(f"  Success Rate: {eval_metrics['success_rate']:.1%}")
    print(f"  Mean Cost: ${eval_metrics['energy_costs']['mean']:.2f}")
    
    # Run one episode for demonstration
    print("\nRunning demonstration episode...")
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(20):
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        
        print(f"Step {step+1:2d}: Storage={info['storage_level']:5.1f}kW, "
              f"Demand={info['demand']:4.1f}kW, Generation={info['generation']:4.1f}kW, "
              f"Reward={reward:6.2f}")
        
        state = next_state
        if done:
            break
    
    print(f"\nEpisode completed! Total reward: {total_reward:.2f}")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
