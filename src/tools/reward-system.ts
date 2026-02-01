/**
 * Reward System - Visual metrics and aesthetic scoring for RL training
 *
 * Implements multi-component reward function based on:
 * - Visual entropy (complexity)
 * - Color harmony
 * - Motion coherence
 * - Novelty bonus
 * - FPS penalty
 */

import type { FrameMetrics, RewardComponents, NoveltyState } from '../types.js';

// Reward weights (configurable)
const DEFAULT_WEIGHTS = {
  entropy: 0.3,
  color_harmony: 0.2,
  motion_coherence: 0.2,
  novelty: 0.2,
  fps_penalty: 0.1
};

// Novelty tracking state
let noveltyState: NoveltyState = {
  visited_regions: [],
  visit_counts: new Map(),
  total_visits: 0
};

// Previous frame metrics for motion calculation
let previousMetrics: FrameMetrics | null = null;

/**
 * Calculate entropy score from raw entropy value
 * Optimal entropy is in the mid-range (not too simple, not too noisy)
 */
export function calculateEntropyScore(entropy: number): number {
  // Entropy typically ranges from 0 to ~8 for 8-bit images
  // We want to reward mid-range entropy (interesting complexity)
  const optimalEntropy = 5.5;
  const tolerance = 2.0;

  const deviation = Math.abs(entropy - optimalEntropy);
  const score = Math.max(0, 1 - (deviation / tolerance));

  return score;
}

/**
 * Calculate color harmony score from color variance
 * Rewards balanced color distribution (not monotone, not chaotic)
 */
export function calculateColorHarmonyScore(colorVariance: number): number {
  // Color variance ranges from 0 (monotone) to very high (chaotic)
  // We reward moderate variance indicating harmonious color schemes
  const optimalVariance = 0.15;
  const tolerance = 0.1;

  const deviation = Math.abs(colorVariance - optimalVariance);
  const score = Math.max(0, 1 - (deviation / tolerance));

  return score;
}

/**
 * Calculate motion coherence score
 * Rewards smooth, intentional motion over jitter
 */
export function calculateMotionCoherenceScore(
  currentMotion: number,
  previousMotion: number | null
): number {
  if (previousMotion === null) {
    // First frame, assume coherent
    return 0.8;
  }

  // Penalize large sudden changes in motion intensity (jitter)
  const motionDelta = Math.abs(currentMotion - previousMotion);
  const jitterPenalty = Math.min(motionDelta * 2, 1);

  // Reward moderate, consistent motion
  const optimalMotion = 0.3;
  const motionScore = 1 - Math.abs(currentMotion - optimalMotion);

  const coherenceScore = motionScore * (1 - jitterPenalty * 0.5);

  return Math.max(0, Math.min(1, coherenceScore));
}

/**
 * Calculate novelty bonus based on parameter space exploration
 */
export function calculateNoveltyBonus(params: Record<string, number>): number {
  // Create a region key from discretized parameters
  const regionKey = Object.entries(params)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}:${Math.round(v * 10) / 10}`)
    .join('|');

  const visitCount = noveltyState.visit_counts.get(regionKey) || 0;
  noveltyState.visit_counts.set(regionKey, visitCount + 1);
  noveltyState.total_visits++;

  // Higher bonus for less-visited regions
  if (visitCount === 0) {
    return 1.0; // First visit bonus
  }

  // Diminishing returns for repeat visits
  const noveltyScore = 1 / (1 + Math.log(visitCount + 1));

  return noveltyScore;
}

/**
 * Calculate FPS penalty
 * Penalize parameter combinations that hurt performance
 */
export function calculateFpsPenalty(fps: number, targetFps: number = 30): number {
  if (fps >= targetFps) {
    return 0; // No penalty if meeting target
  }

  // Linear penalty for low FPS
  const penalty = (targetFps - fps) / targetFps;

  return Math.min(1, penalty);
}

/**
 * Calculate total reward from all components
 */
export function calculateReward(
  metrics: FrameMetrics,
  params: Record<string, number>,
  weights: typeof DEFAULT_WEIGHTS = DEFAULT_WEIGHTS
): RewardComponents {
  // Individual scores
  const entropyScore = calculateEntropyScore(metrics.entropy);
  const colorHarmony = calculateColorHarmonyScore(metrics.color_variance);

  const previousMotion = previousMetrics?.motion_intensity ?? null;
  const motionCoherence = calculateMotionCoherenceScore(
    metrics.motion_intensity,
    previousMotion
  );

  const noveltyBonus = calculateNoveltyBonus(params);
  const fpsPenalty = calculateFpsPenalty(metrics.fps);

  // Update previous metrics for next calculation
  previousMetrics = { ...metrics };

  // Weighted total reward
  const totalReward =
    weights.entropy * entropyScore +
    weights.color_harmony * colorHarmony +
    weights.motion_coherence * motionCoherence +
    weights.novelty * noveltyBonus -
    weights.fps_penalty * fpsPenalty;

  return {
    entropy_score: entropyScore,
    color_harmony: colorHarmony,
    motion_coherence: motionCoherence,
    novelty_bonus: noveltyBonus,
    fps_penalty: fpsPenalty,
    total_reward: Math.max(0, Math.min(1, totalReward))
  };
}

/**
 * Reset novelty tracking state
 */
export function resetNoveltyState(): void {
  noveltyState = {
    visited_regions: [],
    visit_counts: new Map(),
    total_visits: 0
  };
  previousMetrics = null;
}

/**
 * Get current novelty state statistics
 */
export function getNoveltyStats(): {
  unique_regions: number;
  total_visits: number;
  exploration_ratio: number;
} {
  const uniqueRegions = noveltyState.visit_counts.size;
  const totalVisits = noveltyState.total_visits;
  const explorationRatio = totalVisits > 0 ? uniqueRegions / totalVisits : 0;

  return {
    unique_regions: uniqueRegions,
    total_visits: totalVisits,
    exploration_ratio: explorationRatio
  };
}

/**
 * Check if a parameter set produces "interesting" visuals
 * (high reward threshold)
 */
export function isInteresting(rewardComponents: RewardComponents, threshold: number = 0.7): boolean {
  return rewardComponents.total_reward >= threshold;
}

/**
 * Get reward breakdown as human-readable string
 */
export function formatRewardBreakdown(components: RewardComponents): string {
  return `
Reward Breakdown:
  Entropy Score:      ${(components.entropy_score * 100).toFixed(1)}%
  Color Harmony:      ${(components.color_harmony * 100).toFixed(1)}%
  Motion Coherence:   ${(components.motion_coherence * 100).toFixed(1)}%
  Novelty Bonus:      ${(components.novelty_bonus * 100).toFixed(1)}%
  FPS Penalty:       -${(components.fps_penalty * 100).toFixed(1)}%
  ----------------------------
  Total Reward:       ${(components.total_reward * 100).toFixed(1)}%
`.trim();
}

/**
 * Update reward weights dynamically
 */
export function updateWeights(newWeights: Partial<typeof DEFAULT_WEIGHTS>): typeof DEFAULT_WEIGHTS {
  return { ...DEFAULT_WEIGHTS, ...newWeights };
}
