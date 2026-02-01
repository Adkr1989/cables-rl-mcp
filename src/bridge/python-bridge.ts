/**
 * Python Bridge - Spawns and communicates with Python RL processes
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { fileURLToPath } from 'url';
import type { PythonRequest, PythonResponse, AgentConfig, TrainingStepResult, FrameMetrics } from '../types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class PythonBridge extends EventEmitter {
  private process: ChildProcess | null = null;
  private pendingRequests: Map<string, { resolve: (value: unknown) => void; reject: (error: Error) => void }> = new Map();
  private buffer: string = '';
  private pythonPath: string;
  private scriptPath: string;
  private initialized: boolean = false;

  constructor() {
    super();
    this.pythonPath = process.env.PYTHON_PATH || 'python';
    this.scriptPath = path.resolve(__dirname, '../../python/agent_wrapper.py');
  }

  /**
   * Start the Python RL process
   */
  async start(): Promise<void> {
    if (this.process) {
      throw new Error('Python bridge already started');
    }

    return new Promise((resolve, reject) => {
      this.process = spawn(this.pythonPath, [this.scriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1'
        }
      });

      this.process.stdout?.on('data', (data: Buffer) => {
        this.handleStdout(data.toString());
      });

      this.process.stderr?.on('data', (data: Buffer) => {
        console.error('[Python stderr]:', data.toString());
        this.emit('error', data.toString());
      });

      this.process.on('error', (error) => {
        console.error('[Python process error]:', error);
        this.emit('process-error', error);
        reject(error);
      });

      this.process.on('exit', (code) => {
        console.log(`[Python process exited with code ${code}]`);
        this.initialized = false;
        this.process = null;
        this.emit('exit', code);
      });

      // Wait for ready signal
      const timeout = setTimeout(() => {
        reject(new Error('Python bridge initialization timeout'));
      }, 30000);

      this.once('ready', () => {
        clearTimeout(timeout);
        this.initialized = true;
        resolve();
      });
    });
  }

  /**
   * Handle stdout data from Python process
   */
  private handleStdout(data: string): void {
    this.buffer += data;

    // Process complete JSON messages (newline-delimited)
    const lines = this.buffer.split('\n');
    this.buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;

      try {
        const message = JSON.parse(line);

        if (message.type === 'ready') {
          this.emit('ready');
        } else if (message.type === 'response') {
          const pending = this.pendingRequests.get(message.id);
          if (pending) {
            this.pendingRequests.delete(message.id);
            if (message.success) {
              pending.resolve(message.result);
            } else {
              pending.reject(new Error(message.error || 'Unknown error'));
            }
          }
        } else if (message.type === 'event') {
          this.emit(message.event, message.data);
        }
      } catch (e) {
        // Log parsing errors but continue processing
        console.error('[Python bridge parse error]:', e, 'Line:', line);
      }
    }
  }

  /**
   * Send a request to the Python process
   */
  private async sendRequest(method: string, params: Record<string, unknown>): Promise<unknown> {
    if (!this.process || !this.initialized) {
      throw new Error('Python bridge not initialized');
    }

    const id = uuidv4();
    const request: PythonRequest = { id, method, params };

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });

      const timeout = setTimeout(() => {
        this.pendingRequests.delete(id);
        reject(new Error(`Request ${method} timed out`));
      }, 60000);

      this.process?.stdin?.write(JSON.stringify(request) + '\n', (error) => {
        if (error) {
          clearTimeout(timeout);
          this.pendingRequests.delete(id);
          reject(error);
        }
      });

      // Clear timeout on response (handled in resolve/reject)
      const originalResolve = this.pendingRequests.get(id)?.resolve;
      const originalReject = this.pendingRequests.get(id)?.reject;

      if (originalResolve && originalReject) {
        this.pendingRequests.set(id, {
          resolve: (value) => { clearTimeout(timeout); originalResolve(value); },
          reject: (error) => { clearTimeout(timeout); originalReject(error); }
        });
      }
    });
  }

  /**
   * Initialize RL agent
   */
  async initAgent(config: AgentConfig): Promise<{ agent_id: string }> {
    const result = await this.sendRequest('init_agent', { config });
    return result as { agent_id: string };
  }

  /**
   * Execute a training step
   */
  async trainingStep(params: {
    observation: FrameMetrics;
    num_steps: number;
  }): Promise<TrainingStepResult> {
    const result = await this.sendRequest('training_step', params);
    return result as TrainingStepResult;
  }

  /**
   * Get action from agent given current observation
   */
  async getAction(observation: FrameMetrics): Promise<Record<string, number>> {
    const result = await this.sendRequest('get_action', { observation });
    return result as Record<string, number>;
  }

  /**
   * Update agent with reward
   */
  async updateReward(reward: number, done: boolean): Promise<void> {
    await this.sendRequest('update_reward', { reward, done });
  }

  /**
   * Calculate reward from frame metrics
   */
  async calculateReward(metrics: FrameMetrics): Promise<number> {
    const result = await this.sendRequest('calculate_reward', { metrics });
    return result as number;
  }

  /**
   * Save agent checkpoint
   */
  async saveCheckpoint(path: string): Promise<void> {
    await this.sendRequest('save_checkpoint', { path });
  }

  /**
   * Load agent checkpoint
   */
  async loadCheckpoint(path: string): Promise<void> {
    await this.sendRequest('load_checkpoint', { path });
  }

  /**
   * Stop the Python process
   */
  async stop(): Promise<void> {
    if (this.process) {
      try {
        await this.sendRequest('shutdown', {});
      } catch (e) {
        // Ignore shutdown errors
      }
      this.process.kill();
      this.process = null;
      this.initialized = false;
    }
  }

  /**
   * Check if bridge is ready
   */
  isReady(): boolean {
    return this.initialized && this.process !== null;
  }
}

// Singleton instance
let bridgeInstance: PythonBridge | null = null;

export function getPythonBridge(): PythonBridge {
  if (!bridgeInstance) {
    bridgeInstance = new PythonBridge();
  }
  return bridgeInstance;
}
