#!/usr/bin/env python3
"""
Light_RL Agent Wrapper

Provides a JSON-RPC interface over stdin/stdout for communication with
the Node.js MCP server. Wraps Light_RL agents for visual parameter optimization.
"""

import sys
import json
import uuid
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import traceback

# Try to import Light_RL components
try:
    from light_rl.agents import PPOAgent, SACAgent, TD3Agent
    from light_rl.common import ReplayBuffer
    LIGHT_RL_AVAILABLE = True
except ImportError:
    LIGHT_RL_AVAILABLE = False
    print(json.dumps({"type": "warning", "message": "Light_RL not installed, using mock agent"}), flush=True)


@dataclass
class ParameterRange:
    """Definition of a parameter's valid range"""
    op_name: str
    param_name: str
    min_val: float
    max_val: float
    param_type: str = "continuous"
    step: Optional[float] = None


class MockAgent:
    """Mock agent for testing when Light_RL is not available"""

    def __init__(self, obs_dim: int, action_dim: int, action_ranges: list):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_ranges = action_ranges
        self.step_count = 0

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Return random actions within valid ranges"""
        actions = []
        for range_info in self.action_ranges:
            action = np.random.uniform(range_info["min"], range_info["max"])
            actions.append(action)
        return np.array(actions)

    def update(self, obs, action, reward, next_obs, done):
        """Mock update - does nothing"""
        self.step_count += 1
        return {"loss": 0.0}

    def save(self, path: str):
        """Save mock checkpoint"""
        import json
        with open(path, 'w') as f:
            json.dump({"step_count": self.step_count}, f)

    def load(self, path: str):
        """Load mock checkpoint"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.step_count = data.get("step_count", 0)


class AgentManager:
    """Manages RL agent lifecycle and training"""

    def __init__(self):
        self.agent = None
        self.agent_id: Optional[str] = None
        self.action_space: list = []
        self.observation_dim = 4  # fps, entropy, color_variance, motion_intensity
        self.last_observation: Optional[np.ndarray] = None
        self.last_action: Optional[np.ndarray] = None
        self.episode_rewards: list = []
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.best_params: Dict[str, float] = {}

    def init_agent(self, config: Dict[str, Any]) -> str:
        """Initialize RL agent with given configuration"""
        algorithm = config.get("algorithm", "SAC")
        action_space = config.get("action_space", [])

        self.action_space = action_space
        action_dim = len(action_space)

        # Create action ranges for normalization
        action_ranges = [
            {"min": p["min"], "max": p["max"], "name": f"{p['op_name']}.{p['param_name']}"}
            for p in action_space
        ]

        if LIGHT_RL_AVAILABLE:
            # Initialize Light_RL agent
            if algorithm == "PPO":
                self.agent = PPOAgent(
                    obs_dim=self.observation_dim,
                    action_dim=action_dim,
                    lr=config.get("learning_rate", 3e-4)
                )
            elif algorithm == "SAC":
                self.agent = SACAgent(
                    obs_dim=self.observation_dim,
                    action_dim=action_dim,
                    lr=config.get("learning_rate", 3e-4)
                )
            elif algorithm == "TD3":
                self.agent = TD3Agent(
                    obs_dim=self.observation_dim,
                    action_dim=action_dim,
                    lr=config.get("learning_rate", 3e-4)
                )
            else:
                self.agent = SACAgent(
                    obs_dim=self.observation_dim,
                    action_dim=action_dim,
                    lr=config.get("learning_rate", 3e-4)
                )
        else:
            # Use mock agent
            self.agent = MockAgent(
                obs_dim=self.observation_dim,
                action_dim=action_dim,
                action_ranges=action_ranges
            )

        self.agent_id = str(uuid.uuid4())
        self.episode_rewards = []
        self.total_steps = 0

        return self.agent_id

    def get_action(self, observation: Dict[str, float]) -> Dict[str, float]:
        """Get action from agent given observation"""
        if self.agent is None:
            raise ValueError("Agent not initialized")

        # Convert observation dict to numpy array
        obs = np.array([
            observation.get("fps", 60) / 60.0,  # Normalize FPS
            observation.get("entropy", 5.0) / 8.0,  # Normalize entropy
            observation.get("color_variance", 0.15),
            observation.get("motion_intensity", 0.3)
        ], dtype=np.float32)

        # Get action from agent
        action = self.agent.get_action(obs)

        # Store for learning update
        self.last_observation = obs
        self.last_action = action

        # Convert action to parameter dict
        params = {}
        for i, param_def in enumerate(self.action_space):
            key = f"{param_def['op_name']}.{param_def['param_name']}"
            # Scale action from [-1, 1] or [0, 1] to parameter range
            min_val = param_def["min"]
            max_val = param_def["max"]

            if hasattr(self.agent, 'action_scale'):
                # Light_RL agent outputs in [-1, 1]
                scaled_value = min_val + (action[i] + 1) * 0.5 * (max_val - min_val)
            else:
                # Mock agent outputs in [min, max] directly
                scaled_value = action[i]

            params[key] = float(np.clip(scaled_value, min_val, max_val))

        return params

    def update_reward(self, reward: float, done: bool):
        """Update agent with reward signal"""
        if self.agent is None or self.last_observation is None:
            return

        self.episode_rewards.append(reward)
        self.total_steps += 1

        # Track best reward
        if reward > self.best_reward:
            self.best_reward = reward

        # For Light_RL agents, we need next observation
        # Using same observation as placeholder (will be updated next step)
        if hasattr(self.agent, 'update'):
            self.agent.update(
                self.last_observation,
                self.last_action,
                reward,
                self.last_observation,  # Will be replaced
                done
            )

    def training_step(self, observation: Dict[str, float], num_steps: int) -> Dict[str, Any]:
        """Execute training steps and return results"""
        results = {
            "episode_reward": sum(self.episode_rewards[-num_steps:]) if self.episode_rewards else 0,
            "best_params": self.best_params,
            "screenshot": "",
            "step_count": self.total_steps
        }

        # Get action for current observation
        params = self.get_action(observation)
        self.best_params = params
        results["best_params"] = params

        return results

    def save_checkpoint(self, path: str):
        """Save agent checkpoint"""
        if self.agent is not None:
            self.agent.save(path)

    def load_checkpoint(self, path: str):
        """Load agent from checkpoint"""
        if self.agent is not None:
            self.agent.load(path)


class JSONRPCHandler:
    """Handles JSON-RPC messages over stdin/stdout"""

    def __init__(self):
        self.manager = AgentManager()
        self.methods = {
            "init_agent": self.handle_init_agent,
            "get_action": self.handle_get_action,
            "update_reward": self.handle_update_reward,
            "training_step": self.handle_training_step,
            "calculate_reward": self.handle_calculate_reward,
            "save_checkpoint": self.handle_save_checkpoint,
            "load_checkpoint": self.handle_load_checkpoint,
            "shutdown": self.handle_shutdown
        }

    def handle_init_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        config = params.get("config", {})
        agent_id = self.manager.init_agent(config)
        return {"agent_id": agent_id}

    def handle_get_action(self, params: Dict[str, Any]) -> Dict[str, float]:
        observation = params.get("observation", {})
        return self.manager.get_action(observation)

    def handle_update_reward(self, params: Dict[str, Any]) -> Dict[str, bool]:
        reward = params.get("reward", 0.0)
        done = params.get("done", False)
        self.manager.update_reward(reward, done)
        return {"success": True}

    def handle_training_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        observation = params.get("observation", {})
        num_steps = params.get("num_steps", 1)
        return self.manager.training_step(observation, num_steps)

    def handle_calculate_reward(self, params: Dict[str, Any]) -> float:
        # Delegate to reward functions
        from reward_functions import calculate_total_reward
        metrics = params.get("metrics", {})
        return calculate_total_reward(metrics)

    def handle_save_checkpoint(self, params: Dict[str, Any]) -> Dict[str, bool]:
        path = params.get("path", "checkpoint.pt")
        self.manager.save_checkpoint(path)
        return {"success": True}

    def handle_load_checkpoint(self, params: Dict[str, Any]) -> Dict[str, bool]:
        path = params.get("path", "checkpoint.pt")
        self.manager.load_checkpoint(path)
        return {"success": True}

    def handle_shutdown(self, params: Dict[str, Any]) -> Dict[str, bool]:
        return {"success": True}

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single JSON-RPC request"""
        request_id = request.get("id", "")
        method = request.get("method", "")
        params = request.get("params", {})

        try:
            if method not in self.methods:
                return {
                    "id": request_id,
                    "type": "response",
                    "success": False,
                    "error": f"Unknown method: {method}"
                }

            result = self.methods[method](params)

            return {
                "id": request_id,
                "type": "response",
                "success": True,
                "result": result
            }

        except Exception as e:
            return {
                "id": request_id,
                "type": "response",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def run(self):
        """Main loop - read from stdin, process, write to stdout"""
        # Send ready signal
        print(json.dumps({"type": "ready"}), flush=True)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.process_request(request)
                print(json.dumps(response), flush=True)

                # Check for shutdown
                if request.get("method") == "shutdown":
                    break

            except json.JSONDecodeError as e:
                error_response = {
                    "type": "response",
                    "success": False,
                    "error": f"Invalid JSON: {e}"
                }
                print(json.dumps(error_response), flush=True)


def main():
    handler = JSONRPCHandler()
    handler.run()


if __name__ == "__main__":
    main()
