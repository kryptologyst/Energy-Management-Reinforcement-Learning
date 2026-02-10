"""Interactive Streamlit Demo for Energy Management RL.

This demo provides an interactive interface to visualize and test
energy management reinforcement learning policies.

WARNING: This is a research/educational implementation. NOT FOR PRODUCTION USE
in real energy management systems.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.environments.energy_management import EnergyManagementEnv
from src.algorithms.modern_rl import DQNAgent, SACAgent
from src.utils.config import load_config, get_device_config, setup_reproducibility


def load_trained_agent(agent_type: str, model_path: str, device: torch.device):
    """Load a trained agent from file."""
    env = EnergyManagementEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    if agent_type.lower() == 'dqn':
        agent = DQNAgent(state_dim, action_dim, device=device)
    elif agent_type.lower() == 'sac':
        agent = SACAgent(state_dim, action_dim, device=device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.load(model_path)
    return agent, env


def run_episode(agent, env, max_steps=100, render=False):
    """Run a single episode and collect data."""
    state, _ = env.reset()
    episode_data = []
    
    for step in range(max_steps):
        # Select action
        if isinstance(agent, DQNAgent):
            action = agent.select_action(state, training=False)
            action_array = np.array([action])
        else:
            action_array = agent.select_action(state, training=False)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action_array)
        done = terminated or truncated
        
        # Store data
        episode_data.append({
            'step': step,
            'state': state.copy(),
            'action': action_array.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'storage_level': info.get('storage_level', 0),
            'demand': info.get('demand', 0),
            'generation': info.get('generation', 0),
            'grid_price': info.get('grid_price', 0),
            'energy_cost': info.get('energy_cost', 0),
            'hour_of_day': info.get('hour_of_day', 0),
        })
        
        state = next_state
        if done:
            break
    
    return episode_data


def create_energy_plots(episode_data):
    """Create interactive plots for energy management visualization."""
    df = pd.DataFrame(episode_data)
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Energy Storage Level', 'Energy Demand vs Generation',
            'Grid Price', 'Energy Cost',
            'Actions Taken', 'Reward Over Time'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Storage level
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['storage_level'], name='Storage Level', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Demand vs Generation
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['demand'], name='Demand', line=dict(color='red')),
        row=1, col=2, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['generation'], name='Generation', line=dict(color='green')),
        row=1, col=2, secondary_y=True
    )
    
    # Grid price
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['grid_price'], name='Grid Price', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Energy cost
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['energy_cost'], name='Energy Cost', line=dict(color='purple')),
        row=2, col=2
    )
    
    # Actions
    actions_df = pd.DataFrame(df['action'].tolist(), columns=['Charge', 'Discharge', 'Grid Buy', 'Grid Sell'])
    for i, col in enumerate(actions_df.columns):
        fig.add_trace(
            go.Scatter(x=df['step'], y=actions_df[col], name=f'Action: {col}', line=dict(width=2)),
            row=3, col=1
        )
    
    # Reward
    fig.add_trace(
        go.Scatter(x=df['step'], y=df['reward'], name='Reward', line=dict(color='gold')),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        title_text="Energy Management RL Agent Performance",
        showlegend=True
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Storage Level (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Demand (kW)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Generation (kW)", row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Price ($/kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Cost ($)", row=2, col=2)
    fig.update_yaxes(title_text="Action Value", row=3, col=1)
    fig.update_yaxes(title_text="Reward", row=3, col=2)
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Energy Management RL Demo",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ Energy Management Reinforcement Learning Demo")
    
    # Safety warning
    st.warning(
        "⚠️ **WARNING**: This is a research/educational simulation. "
        "NOT FOR PRODUCTION USE in real energy management systems. "
        "Real systems require extensive safety validation and regulatory compliance."
    )
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Agent selection
    agent_type = st.sidebar.selectbox("Agent Type", ["SAC", "DQN"])
    model_path = st.sidebar.text_input("Model Path", "logs/final_model.pth")
    
    # Episode parameters
    max_steps = st.sidebar.slider("Max Steps per Episode", 50, 200, 100)
    num_episodes = st.sidebar.slider("Number of Episodes", 1, 10, 3)
    
    # Environment parameters
    st.sidebar.header("Environment Parameters")
    max_storage = st.sidebar.slider("Max Storage Capacity (kWh)", 50, 200, 100)
    max_demand = st.sidebar.slider("Max Demand (kW)", 25, 100, 50)
    max_generation = st.sidebar.slider("Max Generation (kW)", 40, 120, 80)
    price_variability = st.sidebar.slider("Price Variability", 0.1, 0.5, 0.3)
    
    # Run button
    if st.sidebar.button("🚀 Run Simulation"):
        try:
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load agent
            with st.spinner("Loading trained agent..."):
                agent, env = load_trained_agent(agent_type.lower(), model_path, device)
            
            # Update environment parameters
            env.max_storage_capacity = max_storage
            env.max_demand = max_demand
            env.max_generation = max_generation
            env.price_variability = price_variability
            
            st.success(f"✅ Loaded {agent_type} agent successfully!")
            
            # Run episodes
            all_episodes_data = []
            episode_rewards = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for episode in range(num_episodes):
                status_text.text(f"Running episode {episode + 1}/{num_episodes}...")
                
                episode_data = run_episode(agent, env, max_steps)
                all_episodes_data.extend(episode_data)
                episode_rewards.append(sum([step['reward'] for step in episode_data]))
                
                progress_bar.progress((episode + 1) / num_episodes)
            
            status_text.text("✅ Simulation completed!")
            
            # Display results
            st.header("📊 Results")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Reward", f"{np.mean(episode_rewards):.2f}")
            with col2:
                st.metric("Total Episodes", num_episodes)
            with col3:
                st.metric("Average Steps", f"{len(all_episodes_data) / num_episodes:.1f}")
            with col4:
                total_cost = sum([step['energy_cost'] for step in all_episodes_data])
                st.metric("Total Energy Cost", f"${total_cost:.2f}")
            
            # Create and display plots
            st.header("📈 Performance Visualization")
            
            # Episode comparison
            episode_df = pd.DataFrame({
                'Episode': range(1, num_episodes + 1),
                'Total Reward': episode_rewards
            })
            
            fig_episodes = px.bar(episode_df, x='Episode', y='Total Reward', 
                                 title='Reward per Episode')
            st.plotly_chart(fig_episodes, use_container_width=True)
            
            # Detailed plots for first episode
            if all_episodes_data:
                st.header("🔍 Detailed Episode Analysis")
                
                # Show first episode data
                first_episode_steps = max_steps
                first_episode_data = all_episodes_data[:first_episode_steps]
                
                detailed_fig = create_energy_plots(first_episode_data)
                st.plotly_chart(detailed_fig, use_container_width=True)
                
                # Action distribution
                st.subheader("🎯 Action Distribution")
                actions_df = pd.DataFrame([step['action'] for step in first_episode_data])
                actions_df.columns = ['Charge Rate', 'Discharge Rate', 'Grid Buy Rate', 'Grid Sell Rate']
                
                fig_actions = px.histogram(actions_df, 
                                          title='Action Value Distribution',
                                          labels={'value': 'Action Value', 'count': 'Frequency'})
                st.plotly_chart(fig_actions, use_container_width=True)
                
                # State evolution
                st.subheader("📊 State Evolution")
                states_df = pd.DataFrame([step['state'] for step in first_episode_data])
                states_df.columns = ['Storage Level', 'Demand', 'Generation', 'Price Deviation', 'Hour']
                
                fig_states = px.line(states_df, title='State Variables Over Time')
                st.plotly_chart(fig_states, use_container_width=True)
            
        except FileNotFoundError:
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please train a model first using the training script.")
        except Exception as e:
            st.error(f"❌ Error running simulation: {str(e)}")
    
    # Information section
    st.header("ℹ️ About This Demo")
    
    st.markdown("""
    This interactive demo showcases a reinforcement learning agent trained to optimize energy management in a microgrid system.
    
    **Key Features:**
    - **Real-time Visualization**: See how the agent makes decisions in real-time
    - **Multiple Algorithms**: Compare DQN and SAC agents
    - **Configurable Environment**: Adjust environment parameters to test different scenarios
    - **Comprehensive Metrics**: View detailed performance statistics and visualizations
    
    **Environment Components:**
    - **Energy Storage**: Battery system with configurable capacity
    - **Renewable Generation**: Solar and wind energy sources
    - **Variable Demand**: Time-varying energy consumption
    - **Grid Connection**: Buy/sell energy at dynamic prices
    
    **Agent Actions:**
    - **Charge Rate**: How much energy to store
    - **Discharge Rate**: How much energy to release
    - **Grid Buy Rate**: How much energy to purchase from grid
    - **Grid Sell Rate**: How much energy to sell to grid
    
    **Training**: Use the training script to train your own agents:
    ```bash
    python scripts/train.py --agent-type sac --num-episodes 2000
    ```
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer**: This is a research simulation for educational purposes. "
        "Not suitable for production energy management systems."
    )


if __name__ == "__main__":
    main()
