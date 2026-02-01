/**
 * Cables.gl Browser Control - Automation for visual shader system
 *
 * Uses Playwright MCP for browser automation to control Cables.gl patches.
 * Provides parameter injection, frame capture, and patch management.
 */

import { Tool } from '@modelcontextprotocol/sdk/types.js';
import type { PatchState, FrameMetrics } from '../types.js';
import fs from 'fs/promises';
import path from 'path';

// Current patch state
let currentPatch: PatchState | null = null;
let lastFrameTime = Date.now();
let frameCount = 0;
let currentFps = 60;

// Output directory for screenshots
const OUTPUT_DIR = process.env.OUTPUT_DIR || './output';

/**
 * Tool definitions for Cables.gl control
 */
export const cablesTools: Tool[] = [
  {
    name: 'cables_load_patch',
    description: 'Load a Cables.gl patch in the browser. Supports both cables.gl URLs and local patch files.',
    inputSchema: {
      type: 'object',
      properties: {
        patch_url: {
          type: 'string',
          description: 'URL to the Cables.gl patch (e.g., https://cables.gl/edit/xHwYjG or exported patch URL)'
        },
        headless: {
          type: 'boolean',
          description: 'Run browser without visible window (default: false)',
          default: false
        }
      },
      required: ['patch_url']
    }
  },
  {
    name: 'cables_set_parameter',
    description: 'Set a parameter value in the current Cables.gl patch. The parameter will be modified in real-time.',
    inputSchema: {
      type: 'object',
      properties: {
        op_name: {
          type: 'string',
          description: 'Name of the operator containing the parameter (e.g., "CustomShader_v2")'
        },
        param_name: {
          type: 'string',
          description: 'Name of the parameter to modify (e.g., "speed", "zoom", "colorShift")'
        },
        value: {
          oneOf: [
            { type: 'number' },
            { type: 'string' }
          ],
          description: 'New value for the parameter'
        }
      },
      required: ['op_name', 'param_name', 'value']
    }
  },
  {
    name: 'cables_get_frame_metrics',
    description: 'Capture the current frame and compute visual metrics for reward calculation. Returns entropy, color variance, motion intensity, and FPS.',
    inputSchema: {
      type: 'object',
      properties: {
        save_screenshot: {
          type: 'boolean',
          description: 'Whether to save a screenshot of the frame (default: true)',
          default: true
        }
      }
    }
  },
  {
    name: 'cables_get_parameters',
    description: 'Get all available parameters from the current Cables.gl patch',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  {
    name: 'cables_batch_set_parameters',
    description: 'Set multiple parameters at once for efficient RL control',
    inputSchema: {
      type: 'object',
      properties: {
        parameters: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              op_name: { type: 'string' },
              param_name: { type: 'string' },
              value: { oneOf: [{ type: 'number' }, { type: 'string' }] }
            },
            required: ['op_name', 'param_name', 'value']
          },
          description: 'Array of parameter updates to apply'
        }
      },
      required: ['parameters']
    }
  }
];

/**
 * JavaScript code to inject into Cables.gl page for parameter control
 */
export const cablesInjectionScript = `
(function() {
  // Wait for CABLES to be available
  function waitForCables(callback) {
    if (window.CABLES && window.CABLES.patch) {
      callback();
    } else {
      setTimeout(() => waitForCables(callback), 100);
    }
  }

  // Expose control functions to window
  window.CablesRL = {
    // Set a parameter value
    setParameter: function(opName, paramName, value) {
      const patch = window.CABLES.patch;
      if (!patch) return { success: false, error: 'No patch loaded' };

      // Find the operator
      const ops = patch.ops || [];
      let targetOp = null;

      for (const op of ops) {
        if (op.name === opName || op.objName === opName ||
            (op.uiAttribs && op.uiAttribs.title === opName)) {
          targetOp = op;
          break;
        }
      }

      if (!targetOp) {
        // Try finding by partial match
        for (const op of ops) {
          const name = op.name || op.objName || (op.uiAttribs && op.uiAttribs.title) || '';
          if (name.toLowerCase().includes(opName.toLowerCase())) {
            targetOp = op;
            break;
          }
        }
      }

      if (!targetOp) {
        return { success: false, error: 'Operator not found: ' + opName };
      }

      // Find the port/parameter
      const ports = [...(targetOp.portsIn || []), ...(targetOp.portsOut || [])];
      let targetPort = null;

      for (const port of ports) {
        if (port.name === paramName || port.title === paramName) {
          targetPort = port;
          break;
        }
      }

      if (!targetPort) {
        return { success: false, error: 'Parameter not found: ' + paramName };
      }

      // Set the value
      targetPort.set(value);
      return { success: true, op: targetOp.name, param: paramName, value: value };
    },

    // Get all parameters
    getParameters: function() {
      const patch = window.CABLES.patch;
      if (!patch) return { success: false, error: 'No patch loaded' };

      const result = {};
      const ops = patch.ops || [];

      for (const op of ops) {
        const opName = op.name || op.objName || (op.uiAttribs && op.uiAttribs.title) || 'unknown';
        const ports = op.portsIn || [];

        if (ports.length > 0) {
          result[opName] = {};
          for (const port of ports) {
            if (port.type === 0 || port.type === 1) { // Number or String
              result[opName][port.name] = {
                value: port.get(),
                type: port.type === 0 ? 'number' : 'string',
                min: port.uiAttribs ? port.uiAttribs.min : undefined,
                max: port.uiAttribs ? port.uiAttribs.max : undefined
              };
            }
          }
          if (Object.keys(result[opName]).length === 0) {
            delete result[opName];
          }
        }
      }

      return { success: true, parameters: result };
    },

    // Get FPS
    getFps: function() {
      if (window.CABLES && window.CABLES.patch && window.CABLES.patch.cgl) {
        return window.CABLES.patch.cgl.profileData.fps || 60;
      }
      return 60;
    },

    // Check if ready
    isReady: function() {
      return window.CABLES && window.CABLES.patch && window.CABLES.patch.cgl;
    }
  };

  waitForCables(() => {
    console.log('[CablesRL] Control interface ready');
  });
})();
`;

/**
 * Generate JavaScript to evaluate in browser for setting parameters
 */
export function generateSetParameterScript(opName: string, paramName: string, value: number | string): string {
  const escapedOpName = opName.replace(/'/g, "\\'");
  const escapedParamName = paramName.replace(/'/g, "\\'");
  const valueStr = typeof value === 'string' ? `'${value.replace(/'/g, "\\'")}'` : value;

  return `
    (() => {
      if (!window.CablesRL) {
        return { success: false, error: 'CablesRL not initialized' };
      }
      return window.CablesRL.setParameter('${escapedOpName}', '${escapedParamName}', ${valueStr});
    })()
  `;
}

/**
 * Generate JavaScript to get all parameters
 */
export function generateGetParametersScript(): string {
  return `
    (() => {
      if (!window.CablesRL) {
        return { success: false, error: 'CablesRL not initialized' };
      }
      return window.CablesRL.getParameters();
    })()
  `;
}

/**
 * Generate JavaScript to get FPS
 */
export function generateGetFpsScript(): string {
  return `
    (() => {
      if (!window.CablesRL) {
        return 60;
      }
      return window.CablesRL.getFps();
    })()
  `;
}

/**
 * Generate JavaScript to check if Cables is ready
 */
export function generateIsReadyScript(): string {
  return `
    (() => {
      return window.CablesRL && window.CablesRL.isReady();
    })()
  `;
}

/**
 * Update patch state
 */
export function updatePatchState(state: Partial<PatchState>): void {
  if (!currentPatch) {
    currentPatch = {
      url: '',
      loaded: false,
      parameters: {},
      fps: 60
    };
  }
  Object.assign(currentPatch, state);
}

/**
 * Get current patch state
 */
export function getPatchState(): PatchState | null {
  return currentPatch;
}

/**
 * Calculate FPS from frame timing
 */
export function calculateFps(): number {
  const now = Date.now();
  const elapsed = now - lastFrameTime;

  if (elapsed >= 1000) {
    currentFps = Math.round((frameCount * 1000) / elapsed);
    frameCount = 0;
    lastFrameTime = now;
  }

  frameCount++;
  return currentFps;
}

/**
 * Generate unique screenshot path
 */
export async function generateScreenshotPath(): Promise<string> {
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  return path.join(OUTPUT_DIR, `frame-${timestamp}.png`);
}

/**
 * Tool handler results type
 */
export interface CablesToolResult {
  content: Array<{ type: 'text'; text: string }>;
  isError?: boolean;
}

/**
 * Process cables tool calls
 * Note: This returns instructions for the MCP server to relay to Playwright MCP
 */
export function processCablesToolCall(name: string, args: Record<string, unknown>): CablesToolResult {
  switch (name) {
    case 'cables_load_patch': {
      const patchUrl = args.patch_url as string;
      const headless = args.headless as boolean || false;

      // Update state
      updatePatchState({
        url: patchUrl,
        loaded: false,
        parameters: {},
        fps: 60
      });

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            action: 'load_patch',
            patch_url: patchUrl,
            headless: headless,
            injection_script: cablesInjectionScript,
            instructions: `
1. Use browser_navigate to go to: ${patchUrl}
2. Wait for page to load with browser_wait_for
3. Use browser_evaluate to inject the CablesRL control script
4. Verify CablesRL.isReady() returns true
            `.trim()
          }, null, 2)
        }]
      };
    }

    case 'cables_set_parameter': {
      const opName = args.op_name as string;
      const paramName = args.param_name as string;
      const value = args.value as number | string;

      const script = generateSetParameterScript(opName, paramName, value);

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            action: 'set_parameter',
            op_name: opName,
            param_name: paramName,
            value: value,
            evaluate_script: script,
            instructions: `Use browser_evaluate with function: ${script}`
          }, null, 2)
        }]
      };
    }

    case 'cables_get_frame_metrics': {
      const saveScreenshot = args.save_screenshot !== false;

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            action: 'get_frame_metrics',
            save_screenshot: saveScreenshot,
            fps_script: generateGetFpsScript(),
            instructions: `
1. Use browser_evaluate to get FPS: ${generateGetFpsScript()}
2. ${saveScreenshot ? 'Use browser_take_screenshot to capture frame' : 'Skip screenshot'}
3. Return metrics for reward calculation
            `.trim()
          }, null, 2)
        }]
      };
    }

    case 'cables_get_parameters': {
      const script = generateGetParametersScript();

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            action: 'get_parameters',
            evaluate_script: script,
            instructions: `Use browser_evaluate with function: ${script}`
          }, null, 2)
        }]
      };
    }

    case 'cables_batch_set_parameters': {
      const parameters = args.parameters as Array<{
        op_name: string;
        param_name: string;
        value: number | string;
      }>;

      const scripts = parameters.map(p => ({
        ...p,
        script: generateSetParameterScript(p.op_name, p.param_name, p.value)
      }));

      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            action: 'batch_set_parameters',
            parameters: scripts,
            instructions: `Execute each parameter script sequentially with browser_evaluate`
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
