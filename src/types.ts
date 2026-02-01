/**
 * Type definitions for Cables-RL MCP Server
 */

// Parameter range for RL action space
export interface ParameterRange {
  op_name: string;
  param_name: string;
  min: number;
  max: number;
  step?: number;
  type: 'continuous' | 'discrete';
}

// Frame metrics returned from visual analysis
export interface FrameMetrics {
  fps: number;
  entropy: number;
  color_variance: number;
  motion_intensity: number;
  screenshot_path: string;
  timestamp: number;
}

// RL agent configuration
export interface AgentConfig {
  algorithm: 'PPO' | 'SAC' | 'TD3';
  action_space: ParameterRange[];
  reward_type: 'aesthetic' | 'novelty' | 'custom';
  learning_rate?: number;
  batch_size?: number;
  gamma?: number;
}

// Training step result
export interface TrainingStepResult {
  episode_reward: number;
  best_params: Record<string, number>;
  screenshot: string;
  step_count: number;
  loss?: number;
}

// Exploration session result
export interface ExplorationResult {
  total_episodes: number;
  best_reward: number;
  best_params: Record<string, number>;
  interesting_frames: string[];
  duration_seconds: number;
}

// Cables.gl patch state
export interface PatchState {
  url: string;
  loaded: boolean;
  parameters: Record<string, Record<string, number | string>>;
  fps: number;
}

// Python bridge message types
export interface PythonRequest {
  id: string;
  method: string;
  params: Record<string, unknown>;
}

export interface PythonResponse {
  id: string;
  success: boolean;
  result?: unknown;
  error?: string;
}

// Reward components for aesthetic scoring
export interface RewardComponents {
  entropy_score: number;
  color_harmony: number;
  motion_coherence: number;
  novelty_bonus: number;
  fps_penalty: number;
  total_reward: number;
}

// Novelty tracking state
export interface NoveltyState {
  visited_regions: number[][];
  visit_counts: Map<string, number>;
  total_visits: number;
}

// Session state for persistence
export interface SessionState {
  session_id: string;
  start_time: number;
  patch_url: string;
  agent_config: AgentConfig;
  total_steps: number;
  best_reward: number;
  best_params: Record<string, number>;
  interesting_frames: string[];
}
