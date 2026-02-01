#!/usr/bin/env node
/**
 * Cables-RL MCP Server
 *
 * Integrates Cables.gl (visual shader system) with Light_RL (reinforcement learning)
 * for autonomous visual parameter optimization and generative art exploration.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ErrorCode,
  McpError,
  CallToolResult,
} from '@modelcontextprotocol/sdk/types.js';

import { cablesTools, processCablesToolCall } from './tools/cables-control.js';
import { rlAgentTools, processRlAgentToolCall } from './tools/rl-agent.js';
import { trainingLoopTools, processTrainingLoopToolCall } from './tools/training-loop.js';
import { getPythonBridge } from './bridge/python-bridge.js';

// Helper to create proper CallToolResult
function createToolResult(content: Array<{ type: 'text'; text: string }>, isError?: boolean): CallToolResult {
  return {
    content,
    isError
  };
}

// Server metadata
const SERVER_NAME = 'cables-rl';
const SERVER_VERSION = '1.0.0';

/**
 * Create and configure the MCP server
 */
function createServer(): Server {
  const server = new Server(
    {
      name: SERVER_NAME,
      version: SERVER_VERSION,
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Combine all tools
  const allTools = [...cablesTools, ...rlAgentTools, ...trainingLoopTools];

  // Handle list tools request
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: allTools,
    };
  });

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request): Promise<CallToolResult> => {
    const { name, arguments: args } = request.params;

    try {
      let result: { content: Array<{ type: 'text'; text: string }>; isError?: boolean };

      // Route to appropriate handler based on tool name prefix
      if (name.startsWith('cables_')) {
        result = processCablesToolCall(name, args as Record<string, unknown>);
      } else if (name.startsWith('rl_')) {
        // Check if it's a training loop tool
        const trainingTools = ['rl_autonomous_explore', 'rl_stop_exploration', 'rl_get_exploration_status', 'rl_resume_session'];
        if (trainingTools.includes(name)) {
          result = await processTrainingLoopToolCall(name, args as Record<string, unknown>);
        } else {
          // Otherwise it's an RL agent tool
          result = await processRlAgentToolCall(name, args as Record<string, unknown>);
        }
      } else {
        throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
      }

      return createToolResult(result.content, result.isError);
    } catch (error) {
      if (error instanceof McpError) {
        throw error;
      }

      const errorMessage = error instanceof Error ? error.message : String(error);
      return createToolResult(
        [{ type: 'text', text: JSON.stringify({ error: errorMessage }) }],
        true
      );
    }
  });

  return server;
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  console.error(`[${SERVER_NAME}] Starting MCP server v${SERVER_VERSION}...`);

  const server = createServer();
  const transport = new StdioServerTransport();

  // Handle shutdown
  process.on('SIGINT', async () => {
    console.error(`[${SERVER_NAME}] Shutting down...`);
    const bridge = getPythonBridge();
    if (bridge.isReady()) {
      await bridge.stop();
    }
    await server.close();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    console.error(`[${SERVER_NAME}] Shutting down...`);
    const bridge = getPythonBridge();
    if (bridge.isReady()) {
      await bridge.stop();
    }
    await server.close();
    process.exit(0);
  });

  // Connect transport
  await server.connect(transport);
  console.error(`[${SERVER_NAME}] Server running on stdio`);
}

// Run the server
main().catch((error) => {
  console.error(`[${SERVER_NAME}] Fatal error:`, error);
  process.exit(1);
});
