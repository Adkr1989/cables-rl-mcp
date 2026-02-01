#!/usr/bin/env python3
"""
Gym Environment for Cables.gl Visual Optimization

A Gymnasium-compatible environment that interfaces with Cables.gl
through the MCP server for RL-based parameter exploration.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import json
import sys

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("Warning: gymnasium/gym not installed", file=sys.stderr)

from reward_functions import (
    calculate_total_reward,
    get_reward_components,
    reset_tracking_state,
    RewardWeights
)


@dataclass
class ParameterSpec:
    """Specification for a controllable parameter"""
    op_name: str
    param_name: str
    min_val: float
    max_val: float
    default_val: Optional[float] = None

    @property
    def range(self) -> float:
        return self.max_val - self.min_val

    def normalize(self, value: float) -> float:
        """Normalize value to [-1, 1]"""
        return 2.0 * (value - self.min_val) / self.range - 1.0

    def denormalize(self, normalized: float) -> float:
        """Convert from [-1, 1] to actual range"""
        return self.min_val + (normalized + 1.0) * 0.5 * self.range


class CablesEnvironment:
    """
    Gym-like environment for Cables.gl visual parameter optimization.

    Observation space: [fps, entropy, color_variance, motion_intensity]
    Action space: Continuous parameters as specified in parameter_specs
    """

    def __init__(self,
                 parameter_specs: List[ParameterSpec],
                 reward_weights: Optional[RewardWeights] = None,
                 max_steps: int = 1000,
                 target_fps: float = 30.0):
        """
        Initialize the environment.

        Args:
            parameter_specs: List of parameters to control
            reward_weights: Custom reward weights
            max_steps: Maximum steps per episode
            target_fps: Target FPS for performance penalty
        """
        self.parameter_specs = parameter_specs
        self.reward_weights = reward_weights or RewardWeights()
        self.max_steps = max_steps
        self.target_fps = target_fps

        self.current_step = 0
        self.current_params: Dict[str, float] = {}
        self.last_observation: Optional[np.ndarray] = None
        self.episode_rewards: List[float] = []

        # Define spaces
        self.observation_dim = 4  # fps, entropy, color_variance, motion_intensity
        self.action_dim = len(parameter_specs)

        if GYM_AVAILABLE:
            # Observation: normalized metrics
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )

            # Action: normalized parameters in [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.action_dim,),
                dtype=np.float32
            )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            Initial observation and info dict
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.episode_rewards = []
        reset_tracking_state()

        # Initialize parameters to defaults or midpoints
        self.current_params = {}
        for spec in self.parameter_specs:
            key = f"{spec.op_name}.{spec.param_name}"
            if spec.default_val is not None:
                self.current_params[key] = spec.default_val
            else:
                self.current_params[key] = (spec.min_val + spec.max_val) / 2

        # Initial observation (placeholder - will be replaced by actual metrics)
        self.last_observation = np.array([1.0, 0.5, 0.15, 0.3], dtype=np.float32)

        info = {
            "params": self.current_params.copy(),
            "step": 0
        }

        return self.last_observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Normalized action in [-1, 1] for each parameter

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Convert action to parameters
        for i, spec in enumerate(self.parameter_specs):
            key = f"{spec.op_name}.{spec.param_name}"
            self.current_params[key] = spec.denormalize(action[i])

        # In actual use, metrics come from Cables.gl
        # Here we return a placeholder that should be replaced
        info = {
            "params": self.current_params.copy(),
            "step": self.current_step,
            "action": action.tolist(),
            "requires_external_metrics": True
        }

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Placeholder observation and reward
        observation = self.last_observation
        reward = 0.0

        return observation, reward, terminated, truncated, info

    def update_metrics(self, metrics: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """
        Update environment with actual metrics from Cables.gl.

        Args:
            metrics: Dict with fps, entropy, color_variance, motion_intensity

        Returns:
            observation, reward
        """
        # Create observation
        observation = np.array([
            metrics.get("fps", 60.0) / 60.0,  # Normalized FPS
            metrics.get("entropy", 5.0) / 8.0,  # Normalized entropy
            metrics.get("color_variance", 0.15),
            metrics.get("motion_intensity", 0.3)
        ], dtype=np.float32)

        self.last_observation = observation

        # Calculate reward
        reward = calculate_total_reward(
            metrics,
            self.current_params,
            self.reward_weights
        )

        self.episode_rewards.append(reward)

        return observation, reward

    def get_action_space_info(self) -> List[Dict[str, Any]]:
        """Get information about the action space"""
        return [
            {
                "op_name": spec.op_name,
                "param_name": spec.param_name,
                "min": spec.min_val,
                "max": spec.max_val,
                "type": "continuous"
            }
            for spec in self.parameter_specs
        ]

    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics for the current episode"""
        if not self.episode_rewards:
            return {
                "mean_reward": 0.0,
                "max_reward": 0.0,
                "min_reward": 0.0,
                "total_reward": 0.0,
                "steps": 0
            }

        return {
            "mean_reward": float(np.mean(self.episode_rewards)),
            "max_reward": float(np.max(self.episode_rewards)),
            "min_reward": float(np.min(self.episode_rewards)),
            "total_reward": float(np.sum(self.episode_rewards)),
            "steps": len(self.episode_rewards)
        }

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render is handled externally by Cables.gl"""
        pass

    def close(self):
        """Clean up resources"""
        pass


def create_anomaly_env() -> CablesEnvironment:
    """
    Create an environment configured for the Anomaly patch.

    Returns:
        Configured CablesEnvironment
    """
    # Example parameters for Anomaly patch
    parameter_specs = [
        ParameterSpec("CustomShader_v2", "speed", 0.0, 2.0, 1.0),
        ParameterSpec("CustomShader_v2", "zoom", 0.1, 5.0, 1.0),
        ParameterSpec("CustomShader_v2", "colorShift", 0.0, 1.0, 0.5),
        ParameterSpec("CustomShader_v2", "intensity", 0.0, 2.0, 1.0),
        ParameterSpec("CustomShader_v2", "complexity", 1.0, 10.0, 5.0),
    ]

    return CablesEnvironment(parameter_specs)


if __name__ == "__main__":
    # Test the environment
    env = create_anomaly_env()

    print("Environment created:")
    print(f"  Observation space: {env.observation_dim} dimensions")
    print(f"  Action space: {env.action_dim} parameters")
    print(f"  Max steps: {env.max_steps}")

    print("\nParameter specs:")
    for spec in env.parameter_specs:
        print(f"  {spec.op_name}.{spec.param_name}: [{spec.min_val}, {spec.max_val}]")

    # Test reset and step
    obs, info = env.reset()
    print(f"\nInitial observation: {obs}")
    print(f"Initial params: {info['params']}")

    # Random action
    if GYM_AVAILABLE:
        action = env.action_space.sample()
    else:
        action = np.random.uniform(-1, 1, env.action_dim)

    obs, reward, term, trunc, info = env.step(action)
    print(f"\nAfter step:")
    print(f"  Action: {action}")
    print(f"  New params: {info['params']}")

    # Simulate metrics update
    fake_metrics = {
        "fps": 55,
        "entropy": 5.2,
        "color_variance": 0.18,
        "motion_intensity": 0.25
    }

    obs, reward = env.update_metrics(fake_metrics)
    print(f"\nAfter metrics update:")
    print(f"  Observation: {obs}")
    print(f"  Reward: {reward:.3f}")

    print(f"\nEpisode stats: {env.get_episode_stats()}")
