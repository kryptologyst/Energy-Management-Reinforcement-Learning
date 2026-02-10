"""Energy Management Environment for Reinforcement Learning.

This module implements a realistic energy management environment that simulates
a microgrid with renewable energy sources, energy storage, and variable demand.
The environment follows the Gymnasium interface and provides continuous
observation and action spaces suitable for modern RL algorithms.

WARNING: This is a research/educational simulation. NOT FOR PRODUCTION USE
in real energy management systems.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Tuple, Optional
import warnings


class EnergyManagementEnv(gym.Env):
    """Energy Management Environment for Microgrid Optimization.
    
    This environment simulates a microgrid with:
    - Renewable energy generation (solar, wind)
    - Energy storage (battery)
    - Variable energy demand
    - Grid connection for buying/selling energy
    
    The agent must optimize energy decisions to minimize costs while
    maintaining grid stability.
    
    Args:
        max_storage_capacity: Maximum battery capacity in kWh
        max_demand: Maximum energy demand in kW
        max_generation: Maximum renewable generation in kW
        price_variability: Price volatility factor (0-1)
        episode_length: Maximum episode length in time steps
        seed: Random seed for reproducibility
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        max_storage_capacity: float = 100.0,
        max_demand: float = 50.0,
        max_generation: float = 80.0,
        price_variability: float = 0.3,
        episode_length: int = 24 * 4,  # 24 hours, 15-min intervals
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        # Environment parameters
        self.max_storage_capacity = max_storage_capacity
        self.max_demand = max_demand
        self.max_generation = max_generation
        self.price_variability = price_variability
        self.episode_length = episode_length
        
        # State variables
        self.current_step = 0
        self.storage_level = 0.0
        self.current_demand = 0.0
        self.current_generation = 0.0
        self.grid_price = 0.0
        
        # Action space: [charge_rate, discharge_rate, grid_buy_rate, grid_sell_rate]
        # Each action is normalized to [0, 1] representing percentage of max rate
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Observation space: [storage_level, demand, generation, price, hour_of_day]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 24.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Initialize random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Cost parameters
        self.storage_efficiency = 0.95
        self.grid_buy_price_base = 0.12  # $/kWh
        self.grid_sell_price_base = 0.08  # $/kWh
        self.storage_degradation_cost = 0.001  # $/kWh
        
        # Safety warnings
        warnings.warn(
            "WARNING: This is a research simulation. NOT FOR PRODUCTION USE "
            "in real energy management systems. Real systems require extensive "
            "safety validation and regulatory compliance.",
            UserWarning,
            stacklevel=2
        )
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Initial observation and info dictionary
        """
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        
        # Reset state variables
        self.current_step = 0
        self.storage_level = self.np_random.uniform(0.2, 0.8) * self.max_storage_capacity
        
        # Generate initial conditions
        self._update_demand()
        self._update_generation()
        self._update_grid_price()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action array [charge_rate, discharge_rate, grid_buy_rate, grid_sell_rate]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip actions to valid range
        action = np.clip(action, 0.0, 1.0)
        
        # Parse actions
        charge_rate = action[0] * self.max_storage_capacity * 0.1  # Max 10% capacity per step
        discharge_rate = action[1] * self.max_storage_capacity * 0.1
        grid_buy_rate = action[2] * self.max_demand * 0.5  # Max 50% of demand
        grid_sell_rate = action[3] * self.max_generation * 0.5  # Max 50% of generation
        
        # Apply energy balance constraints
        net_generation = self.current_generation - self.current_demand
        
        # Calculate actual energy flows
        actual_charge = min(
            charge_rate,
            self.max_storage_capacity - self.storage_level,
            net_generation + grid_buy_rate
        )
        
        actual_discharge = min(
            discharge_rate,
            self.storage_level,
            max(0, self.current_demand - self.current_generation - grid_buy_rate)
        )
        
        actual_grid_buy = min(
            grid_buy_rate,
            max(0, self.current_demand - self.current_generation - actual_discharge)
        )
        
        actual_grid_sell = min(
            grid_sell_rate,
            max(0, self.current_generation - self.current_demand - actual_charge)
        )
        
        # Update storage level
        self.storage_level += actual_charge * self.storage_efficiency - actual_discharge
        
        # Calculate costs and rewards
        energy_cost = (
            actual_grid_buy * self.grid_price +
            actual_discharge * self.storage_degradation_cost -
            actual_grid_sell * self.grid_price * 0.8  # Sell at 80% of buy price
        )
        
        # Reward function: minimize costs, penalize constraint violations
        reward = -energy_cost
        
        # Penalty for constraint violations
        if self.storage_level < 0:
            reward -= 10.0  # Severe penalty for negative storage
            self.storage_level = 0.0
        
        if self.storage_level > self.max_storage_capacity:
            reward -= 10.0  # Severe penalty for overcharging
            self.storage_level = self.max_storage_capacity
        
        # Update environment state
        self.current_step += 1
        self._update_demand()
        self._update_generation()
        self._update_grid_price()
        
        # Check termination conditions
        terminated = False
        truncated = self.current_step >= self.episode_length
        
        observation = self._get_observation()
        info = self._get_info()
        info.update({
            "energy_cost": energy_cost,
            "storage_level": self.storage_level,
            "actual_charge": actual_charge,
            "actual_discharge": actual_discharge,
            "actual_grid_buy": actual_grid_buy,
            "actual_grid_sell": actual_grid_sell,
        })
        
        return observation, reward, terminated, truncated, info
    
    def _update_demand(self) -> None:
        """Update energy demand based on time of day and randomness."""
        # Simulate daily demand pattern (higher during day, lower at night)
        hour_of_day = (self.current_step * 15) // 60  # Convert to hours
        base_demand = 0.3 + 0.4 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        base_demand = max(0.1, base_demand)  # Minimum demand
        
        # Add randomness
        demand_noise = self.np_random.normal(0, 0.1)
        self.current_demand = max(0.0, base_demand + demand_noise) * self.max_demand
    
    def _update_generation(self) -> None:
        """Update renewable energy generation."""
        # Simulate solar generation pattern
        hour_of_day = (self.current_step * 15) // 60
        solar_factor = max(0, np.sin(np.pi * (hour_of_day - 6) / 12))  # 6 AM to 6 PM
        
        # Add wind generation (more random)
        wind_factor = self.np_random.uniform(0.2, 0.8)
        
        # Combine solar and wind
        generation_factor = 0.6 * solar_factor + 0.4 * wind_factor
        
        # Add randomness
        generation_noise = self.np_random.normal(0, 0.05)
        self.current_generation = max(0.0, generation_factor + generation_noise) * self.max_generation
    
    def _update_grid_price(self) -> None:
        """Update grid electricity price."""
        # Time-of-use pricing simulation
        hour_of_day = (self.current_step * 15) // 60
        
        # Peak hours (6-10 AM, 6-10 PM) have higher prices
        if hour_of_day in [6, 7, 8, 9, 18, 19, 20, 21]:
            base_price = 1.5
        else:
            base_price = 1.0
        
        # Add price volatility
        price_noise = self.np_random.normal(0, self.price_variability)
        price_multiplier = max(0.5, base_price + price_noise)
        
        self.grid_price = self.grid_buy_price_base * price_multiplier
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        hour_of_day = (self.current_step * 15) % 24
        
        return np.array([
            self.storage_level / self.max_storage_capacity,  # Normalized storage
            self.current_demand / self.max_demand,  # Normalized demand
            self.current_generation / self.max_generation,  # Normalized generation
            (self.grid_price - self.grid_buy_price_base) / self.grid_buy_price_base,  # Price deviation
            hour_of_day,  # Hour of day
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            "step": self.current_step,
            "storage_level": self.storage_level,
            "demand": self.current_demand,
            "generation": self.current_generation,
            "grid_price": self.grid_price,
            "hour_of_day": (self.current_step * 15) % 24,
        }
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Rendered image if mode is 'rgb_array', None otherwise
        """
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Storage: {self.storage_level:.2f} kWh ({self.storage_level/self.max_storage_capacity*100:.1f}%)")
            print(f"Demand: {self.current_demand:.2f} kW")
            print(f"Generation: {self.current_generation:.2f} kW")
            print(f"Grid Price: ${self.grid_price:.3f}/kWh")
            print(f"Net: {self.current_generation - self.current_demand:.2f} kW")
            print("-" * 40)
        
        elif mode == "rgb_array":
            # Create a simple visualization
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Storage bar
            storage_height = self.storage_level / self.max_storage_capacity
            ax.barh(0, storage_height, height=0.3, color='blue', alpha=0.7, label='Storage')
            
            # Demand and generation bars
            ax.barh(1, self.current_demand / self.max_demand, height=0.3, color='red', alpha=0.7, label='Demand')
            ax.barh(2, self.current_generation / self.max_generation, height=0.3, color='green', alpha=0.7, label='Generation')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 2.5)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Storage', 'Demand', 'Generation'])
            ax.set_xlabel('Normalized Level')
            ax.set_title(f'Energy Management - Step {self.current_step}')
            ax.legend()
            
            # Convert to numpy array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            return buf
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass
