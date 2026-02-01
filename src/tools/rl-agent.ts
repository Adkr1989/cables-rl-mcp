/**
 * RL Agent Tools - Python bridge for Light_RL integration
 */

import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { getPythonBridge } from '../bridge/python-bridge.js';
import type { AgentConfig, ParameterRange, TrainingStepResult, FrameMetrics } from '../types.js';

// Current agent state
let currentAgentId: string | null = null;
let currentConfig: AgentConfig | null = null;

/**
 * Tool definitions for RL agent control
 */
export const rlAgentTools: Tool[] = [
  {
    name: 'rl_init_agent',
    description: 'Initialize a Light_RL agent for visual parameter optimization. Supports PPO, SAC, and TD3 algorithms.',
    inputSchema: {
      type: 'object',
      properties: {
        algorithm: {
          type: 'string',
          enum: ['PPO', 'SAC', 'TD3'],
          description: 'RL algorithm to use. SAC recommended for continuous parameters.'
        },
        action_space: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              op_name: { type: 'string', description: 'Cables operator name' },
              param_name: { type: 'string', description: 'Parameter name' },
              min: { type: 'number', description: 'Minimum value' },
              max: { type: 'number', description: 'Maximum value' },
              step: { type: 'number', description: 'Step size for discrete actions' },
              type: {
                type: 'string',
                enum: ['continuous', 'discrete'],
                description: 'Parameter type'
              }
            },
            required: ['op_name', 'param_name', 'min', 'max', 'type']
          },
          description: 'Parameter ranges defining the action space'
        },
        reward_type: {
          type: 'string',
          enum: ['aesthetic', 'novelty', 'custom'],
          description: 'Type of reward function to use'
        },
        learning_rate: {
          type: 'number',
          description: 'Learning rate for the agent (default: 0.0003)'
        },
        batch_size: {
          type: 'number',
          description: 'Batch size for training (default: 64)'
        }
      },
      required: ['algorithm', 'action_space', 'reward_type']
    }
  },
  {
    name: 'rl_training_step',
    description: 'Execute one or more RL training steps. Returns episode reward and best parameters found.',
    inputSchema: {
      type: 'object',
      properties: {
        num_steps: {
          type: 'number',
          description: 'Number of training steps to execute',
          default: 1
        },
        render: {
          type: 'boolean',
          description: 'Whether to capture screenshots during training',
          default: true
        }
      }
    }
  },
  {
    name: 'rl_get_action',
    description: 'Get the next action from the RL agent given current visual state',
    inputSchema: {
      type: 'object',
      properties: {
        metrics: {
          type: 'object',
          properties: {
            fps: { type: 'number' },
            entropy: { type: 'number' },
            color_variance: { type: 'number' },
            motion_intensity: { type: 'number' }
          },
          description: 'Current frame metrics as observation'
        }
      },
      required: ['metrics']
    }
  },
  {
    name: 'rl_update_reward',
    description: 'Send reward signal to the agent after an action',
    inputSchema: {
      type: 'object',
      properties: {
        reward: {
          type: 'number',
          description: 'Reward value for the previous action'
        },
        done: {
          type: 'boolean',
          description: 'Whether the episode is complete',
          default: false
        }
      },
      required: ['reward']
    }
  },
  {
    name: 'rl_save_checkpoint',
    description: 'Save the current agent state to a checkpoint file',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Path to save the checkpoint'
        }
      },
      required: ['path']
    }
  },
  {
    name: 'rl_load_checkpoint',
    description: 'Load an agent from a checkpoint file',
    inputSchema: {
      type: 'object',
      properties: {
        path: {
          type: 'string',
          description: 'Path to the checkpoint file'
        }
      },
      required: ['path']
    }
  },
  {
    name: 'rl_get_status',
    description: 'Get the current status of the RL agent',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  }
];

/**
 * Process RL agent tool calls
 */
export async function processRlAgentToolCall(
  name: string,
  args: Record<string, unknown>
): Promise<{ content: Array<{ type: 'text'; text: string }>; isError?: boolean }> {
  const bridge = getPythonBridge();

  try {
    switch (name) {
      case 'rl_init_agent': {
        // Ensure Python bridge is running
        if (!bridge.isReady()) {
          await bridge.start();
        }

        const config: AgentConfig = {
          algorithm: args.algorithm as 'PPO' | 'SAC' | 'TD3',
          action_space: args.action_space as ParameterRange[],
          reward_type: args.reward_type as 'aesthetic' | 'novelty' | 'custom',
          learning_rate: args.learning_rate as number | undefined,
          batch_size: args.batch_size as number | undefined
        };

        const result = await bridge.initAgent(config);
        currentAgentId = result.agent_id;
        currentConfig = config;

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              agent_id: result.agent_id,
              algorithm: config.algorithm,
              action_space_size: config.action_space.length,
              reward_type: config.reward_type
            }, null, 2)
          }]
        };
      }

      case 'rl_training_step': {
        if (!currentAgentId) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ error: 'No agent initialized. Call rl_init_agent first.' })
            }],
            isError: true
          };
        }

        const numSteps = (args.num_steps as number) || 1;
        const metrics: FrameMetrics = {
          fps: 60,
          entropy: 5.0,
          color_variance: 0.15,
          motion_intensity: 0.3,
          screenshot_path: '',
          timestamp: Date.now()
        };

        const result = await bridge.trainingStep({
          observation: metrics,
          num_steps: numSteps
        });

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              ...result
            }, null, 2)
          }]
        };
      }

      case 'rl_get_action': {
        if (!currentAgentId) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ error: 'No agent initialized' })
            }],
            isError: true
          };
        }

        const metrics = args.metrics as FrameMetrics;
        const action = await bridge.getAction(metrics);

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              action: action
            }, null, 2)
          }]
        };
      }

      case 'rl_update_reward': {
        if (!currentAgentId) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ error: 'No agent initialized' })
            }],
            isError: true
          };
        }

        const reward = args.reward as number;
        const done = (args.done as boolean) || false;

        await bridge.updateReward(reward, done);

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, reward, done })
          }]
        };
      }

      case 'rl_save_checkpoint': {
        if (!currentAgentId) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ error: 'No agent initialized' })
            }],
            isError: true
          };
        }

        const path = args.path as string;
        await bridge.saveCheckpoint(path);

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, checkpoint_path: path })
          }]
        };
      }

      case 'rl_load_checkpoint': {
        if (!bridge.isReady()) {
          await bridge.start();
        }

        const path = args.path as string;
        await bridge.loadCheckpoint(path);

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, loaded_from: path })
          }]
        };
      }

      case 'rl_get_status': {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              agent_initialized: currentAgentId !== null,
              agent_id: currentAgentId,
              config: currentConfig,
              python_bridge_ready: bridge.isReady()
            }, null, 2)
          }]
        };
      }

      default:
        return {
          content: [{
            type: 'text',
            text: `Unknown tool: ${name}`
          }],
          isError: true
        };
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({ error: errorMessage })
      }],
      isError: true
    };
  }
}

/**
 * Get current agent ID
 */
export function getCurrentAgentId(): string | null {
  return currentAgentId;
}

/**
 * Get current agent config
 */
export function getCurrentConfig(): AgentConfig | null {
  return currentConfig;
}
