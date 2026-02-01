/**
 * Training Loop - Autonomous training orchestration for RL visual exploration
 */

import { Tool } from '@modelcontextprotocol/sdk/types.js';
import { v4 as uuidv4 } from 'uuid';
import fs from 'fs/promises';
import path from 'path';
import type { SessionState, ExplorationResult, FrameMetrics, AgentConfig } from '../types.js';
import { calculateReward, isInteresting, resetNoveltyState, getNoveltyStats } from './reward-system.js';
import { getCurrentAgentId, getCurrentConfig } from './rl-agent.js';

// Active session state
let activeSession: SessionState | null = null;
let explorationRunning = false;
let stopRequested = false;

// Default output directory
const OUTPUT_DIR = process.env.OUTPUT_DIR || './output';

/**
 * Tool definitions for training loop control
 */
export const trainingLoopTools: Tool[] = [
  {
    name: 'rl_autonomous_explore',
    description: 'Run an autonomous visual exploration session. The agent will explore the parameter space and save interesting frames.',
    inputSchema: {
      type: 'object',
      properties: {
        duration_minutes: {
          type: 'number',
          description: 'How long to run the exploration (in minutes)',
          default: 5
        },
        save_interesting: {
          type: 'boolean',
          description: 'Whether to save screenshots of high-reward frames',
          default: true
        },
        output_dir: {
          type: 'string',
          description: 'Directory to save interesting frames',
          default: './output/interesting'
        },
        reward_threshold: {
          type: 'number',
          description: 'Minimum reward to consider a frame interesting (0-1)',
          default: 0.7
        },
        steps_per_episode: {
          type: 'number',
          description: 'Number of steps per training episode',
          default: 100
        }
      }
    }
  },
  {
    name: 'rl_stop_exploration',
    description: 'Stop the currently running autonomous exploration session',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'rl_get_exploration_status',
    description: 'Get the status of the current or last exploration session',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'rl_resume_session',
    description: 'Resume a previous exploration session from saved state',
    inputSchema: {
      type: 'object',
      properties: {
        session_id: {
          type: 'string',
          description: 'ID of the session to resume'
        }
      },
      required: ['session_id']
    }
  }
];

/**
 * Create a new exploration session
 */
function createSession(patchUrl: string, config: AgentConfig): SessionState {
  return {
    session_id: uuidv4(),
    start_time: Date.now(),
    patch_url: patchUrl,
    agent_config: config,
    total_steps: 0,
    best_reward: 0,
    best_params: {},
    interesting_frames: []
  };
}

/**
 * Save session state to disk
 */
async function saveSessionState(session: SessionState): Promise<void> {
  const sessionsDir = path.join(OUTPUT_DIR, 'sessions');
  await fs.mkdir(sessionsDir, { recursive: true });

  const sessionPath = path.join(sessionsDir, `${session.session_id}.json`);
  await fs.writeFile(sessionPath, JSON.stringify(session, null, 2));
}

/**
 * Load session state from disk
 */
async function loadSessionState(sessionId: string): Promise<SessionState | null> {
  try {
    const sessionPath = path.join(OUTPUT_DIR, 'sessions', `${sessionId}.json`);
    const data = await fs.readFile(sessionPath, 'utf-8');
    return JSON.parse(data) as SessionState;
  } catch {
    return null;
  }
}

/**
 * Save interesting frame
 */
async function saveInterestingFrame(
  session: SessionState,
  screenshotPath: string,
  reward: number,
  params: Record<string, number>
): Promise<string> {
  const interestingDir = path.join(OUTPUT_DIR, 'interesting', session.session_id);
  await fs.mkdir(interestingDir, { recursive: true });

  const timestamp = Date.now();
  const filename = `frame_${timestamp}_r${(reward * 100).toFixed(0)}.png`;
  const destPath = path.join(interestingDir, filename);

  // Copy screenshot to interesting folder
  try {
    await fs.copyFile(screenshotPath, destPath);
  } catch {
    // If screenshot doesn't exist, just record the metadata
  }

  // Save metadata
  const metadataPath = path.join(interestingDir, `frame_${timestamp}_r${(reward * 100).toFixed(0)}.json`);
  await fs.writeFile(metadataPath, JSON.stringify({
    timestamp,
    reward,
    params,
    screenshot: destPath
  }, null, 2));

  return destPath;
}

/**
 * Process training loop tool calls
 */
export async function processTrainingLoopToolCall(
  name: string,
  args: Record<string, unknown>
): Promise<{ content: Array<{ type: 'text'; text: string }>; isError?: boolean }> {
  switch (name) {
    case 'rl_autonomous_explore': {
      if (explorationRunning) {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              error: 'Exploration already running',
              session_id: activeSession?.session_id
            })
          }],
          isError: true
        };
      }

      const agentId = getCurrentAgentId();
      const config = getCurrentConfig();

      if (!agentId || !config) {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              error: 'No agent initialized. Call rl_init_agent first.',
              instructions: 'Initialize an RL agent with rl_init_agent before starting exploration'
            })
          }],
          isError: true
        };
      }

      const durationMinutes = (args.duration_minutes as number) || 5;
      const saveInteresting = args.save_interesting !== false;
      const outputDir = (args.output_dir as string) || path.join(OUTPUT_DIR, 'interesting');
      const rewardThreshold = (args.reward_threshold as number) || 0.7;
      const stepsPerEpisode = (args.steps_per_episode as number) || 100;

      // Reset novelty tracking
      resetNoveltyState();

      // Create new session
      activeSession = createSession('current_patch', config);
      explorationRunning = true;
      stopRequested = false;

      // Return exploration configuration
      // The actual loop will be orchestrated by the MCP client
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            success: true,
            session_id: activeSession.session_id,
            config: {
              duration_minutes: durationMinutes,
              save_interesting: saveInteresting,
              output_dir: outputDir,
              reward_threshold: rewardThreshold,
              steps_per_episode: stepsPerEpisode
            },
            instructions: `
Autonomous exploration started. Execute the following loop:

1. Call rl_get_action with current frame metrics
2. Apply returned parameters using cables_batch_set_parameters
3. Wait briefly for visual update
4. Call cables_get_frame_metrics to capture new state
5. Calculate reward and call rl_update_reward
6. If reward >= ${rewardThreshold}, save the frame
7. Repeat until ${durationMinutes} minutes elapsed or rl_stop_exploration called
8. Call rl_get_exploration_status for final results
            `.trim()
          }, null, 2)
        }]
      };
    }

    case 'rl_stop_exploration': {
      if (!explorationRunning) {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ message: 'No exploration running' })
          }]
        };
      }

      stopRequested = true;
      explorationRunning = false;

      // Save final session state
      if (activeSession) {
        await saveSessionState(activeSession);
      }

      const noveltyStats = getNoveltyStats();

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            success: true,
            message: 'Exploration stopped',
            session_id: activeSession?.session_id,
            total_steps: activeSession?.total_steps || 0,
            best_reward: activeSession?.best_reward || 0,
            interesting_frames_count: activeSession?.interesting_frames.length || 0,
            novelty_stats: noveltyStats
          }, null, 2)
        }]
      };
    }

    case 'rl_get_exploration_status': {
      const noveltyStats = getNoveltyStats();

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            running: explorationRunning,
            session: activeSession ? {
              session_id: activeSession.session_id,
              start_time: activeSession.start_time,
              duration_seconds: (Date.now() - activeSession.start_time) / 1000,
              total_steps: activeSession.total_steps,
              best_reward: activeSession.best_reward,
              best_params: activeSession.best_params,
              interesting_frames_count: activeSession.interesting_frames.length
            } : null,
            novelty_stats: noveltyStats
          }, null, 2)
        }]
      };
    }

    case 'rl_resume_session': {
      const sessionId = args.session_id as string;
      const loadedSession = await loadSessionState(sessionId);

      if (!loadedSession) {
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ error: `Session not found: ${sessionId}` })
          }],
          isError: true
        };
      }

      activeSession = loadedSession;

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            success: true,
            message: 'Session resumed',
            session: {
              session_id: loadedSession.session_id,
              total_steps: loadedSession.total_steps,
              best_reward: loadedSession.best_reward,
              interesting_frames_count: loadedSession.interesting_frames.length
            }
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
}

/**
 * Update session with step results (called externally)
 */
export async function recordStepResult(
  metrics: FrameMetrics,
  params: Record<string, number>,
  screenshotPath: string
): Promise<{ reward: number; interesting: boolean; saved_path?: string }> {
  if (!activeSession) {
    throw new Error('No active session');
  }

  // Calculate reward
  const rewardComponents = calculateReward(metrics, params);
  const reward = rewardComponents.total_reward;

  // Update session
  activeSession.total_steps++;

  if (reward > activeSession.best_reward) {
    activeSession.best_reward = reward;
    activeSession.best_params = { ...params };
  }

  // Check if interesting
  const interesting = isInteresting(rewardComponents, 0.7);
  let savedPath: string | undefined;

  if (interesting && screenshotPath) {
    savedPath = await saveInterestingFrame(activeSession, screenshotPath, reward, params);
    activeSession.interesting_frames.push(savedPath);
  }

  // Periodically save session state
  if (activeSession.total_steps % 100 === 0) {
    await saveSessionState(activeSession);
  }

  return { reward, interesting, saved_path: savedPath };
}

/**
 * Check if exploration should stop
 */
export function shouldStopExploration(): boolean {
  return stopRequested || !explorationRunning;
}

/**
 * Get active session
 */
export function getActiveSession(): SessionState | null {
  return activeSession;
}
