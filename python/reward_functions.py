#!/usr/bin/env python3
"""
Reward Functions for Visual Aesthetics

Calculates reward scores based on visual metrics for RL training.
Components:
- Entropy (visual complexity)
- Color harmony
- Motion coherence
- Novelty bonus
- FPS penalty
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import hashlib


@dataclass
class RewardWeights:
    """Configurable weights for reward components"""
    entropy: float = 0.3
    color_harmony: float = 0.2
    motion_coherence: float = 0.2
    novelty: float = 0.2
    fps_penalty: float = 0.1


# Global state for tracking
_previous_motion: Optional[float] = None
_visited_regions: Dict[str, int] = defaultdict(int)
_total_visits: int = 0


def calculate_entropy_score(entropy: float, optimal: float = 5.5, tolerance: float = 2.0) -> float:
    """
    Calculate entropy score for visual complexity.

    Optimal entropy is mid-range (not too simple, not noise).
    Entropy typically ranges from 0 to ~8 for 8-bit images.

    Args:
        entropy: Raw entropy value from image
        optimal: Target optimal entropy value
        tolerance: Acceptable deviation from optimal

    Returns:
        Score from 0 to 1
    """
    deviation = abs(entropy - optimal)
    score = max(0.0, 1.0 - (deviation / tolerance))
    return score


def calculate_color_harmony_score(color_variance: float,
                                   optimal: float = 0.15,
                                   tolerance: float = 0.1) -> float:
    """
    Calculate color harmony score from color variance.

    Rewards balanced color distribution (not monotone, not chaotic).

    Args:
        color_variance: Variance of colors in HSV space
        optimal: Target variance for harmonic colors
        tolerance: Acceptable deviation

    Returns:
        Score from 0 to 1
    """
    deviation = abs(color_variance - optimal)
    score = max(0.0, 1.0 - (deviation / tolerance))
    return score


def calculate_motion_coherence_score(current_motion: float,
                                      previous_motion: Optional[float] = None,
                                      optimal_motion: float = 0.3) -> float:
    """
    Calculate motion coherence score.

    Rewards smooth, intentional motion over jitter.

    Args:
        current_motion: Current frame's motion intensity
        previous_motion: Previous frame's motion intensity (None for first frame)
        optimal_motion: Target motion level

    Returns:
        Score from 0 to 1
    """
    global _previous_motion

    if previous_motion is None:
        previous_motion = _previous_motion

    if previous_motion is None:
        # First frame
        _previous_motion = current_motion
        return 0.8

    # Penalize sudden changes (jitter)
    motion_delta = abs(current_motion - previous_motion)
    jitter_penalty = min(motion_delta * 2, 1.0)

    # Reward moderate, consistent motion
    motion_score = 1.0 - abs(current_motion - optimal_motion)

    coherence_score = motion_score * (1.0 - jitter_penalty * 0.5)

    _previous_motion = current_motion
    return max(0.0, min(1.0, coherence_score))


def calculate_novelty_bonus(params: Dict[str, float],
                            discretization: float = 0.1) -> float:
    """
    Calculate novelty bonus based on parameter space exploration.

    Higher bonus for less-visited parameter regions.

    Args:
        params: Current parameter values
        discretization: Resolution for region bucketing

    Returns:
        Score from 0 to 1
    """
    global _visited_regions, _total_visits

    # Create region key from discretized parameters
    sorted_params = sorted(params.items())
    region_parts = [f"{k}:{round(v / discretization) * discretization:.2f}"
                    for k, v in sorted_params]
    region_key = "|".join(region_parts)

    # Hash for memory efficiency
    region_hash = hashlib.md5(region_key.encode()).hexdigest()[:16]

    visit_count = _visited_regions[region_hash]
    _visited_regions[region_hash] += 1
    _total_visits += 1

    if visit_count == 0:
        return 1.0  # First visit bonus

    # Diminishing returns for repeat visits
    novelty_score = 1.0 / (1.0 + np.log(visit_count + 1))
    return novelty_score


def calculate_fps_penalty(fps: float, target_fps: float = 30.0) -> float:
    """
    Calculate FPS penalty for performance impact.

    Args:
        fps: Current frames per second
        target_fps: Minimum acceptable FPS

    Returns:
        Penalty from 0 to 1 (0 = no penalty, 1 = max penalty)
    """
    if fps >= target_fps:
        return 0.0

    penalty = (target_fps - fps) / target_fps
    return min(1.0, penalty)


def calculate_total_reward(metrics: Dict[str, Any],
                           params: Optional[Dict[str, float]] = None,
                           weights: Optional[RewardWeights] = None) -> float:
    """
    Calculate total reward from all components.

    Args:
        metrics: Dict containing fps, entropy, color_variance, motion_intensity
        params: Current parameter values (for novelty calculation)
        weights: Optional custom weights

    Returns:
        Total reward score from 0 to 1
    """
    if weights is None:
        weights = RewardWeights()

    # Extract metrics with defaults
    fps = metrics.get("fps", 60.0)
    entropy = metrics.get("entropy", 5.0)
    color_variance = metrics.get("color_variance", 0.15)
    motion_intensity = metrics.get("motion_intensity", 0.3)

    # Calculate individual scores
    entropy_score = calculate_entropy_score(entropy)
    color_harmony = calculate_color_harmony_score(color_variance)
    motion_coherence = calculate_motion_coherence_score(motion_intensity)
    fps_penalty = calculate_fps_penalty(fps)

    # Novelty bonus (requires params)
    novelty_bonus = 0.5  # Default if no params
    if params is not None:
        novelty_bonus = calculate_novelty_bonus(params)

    # Weighted sum
    total = (
        weights.entropy * entropy_score +
        weights.color_harmony * color_harmony +
        weights.motion_coherence * motion_coherence +
        weights.novelty * novelty_bonus -
        weights.fps_penalty * fps_penalty
    )

    return max(0.0, min(1.0, total))


def get_reward_components(metrics: Dict[str, Any],
                          params: Optional[Dict[str, float]] = None,
                          weights: Optional[RewardWeights] = None) -> Dict[str, float]:
    """
    Get detailed breakdown of all reward components.

    Args:
        metrics: Visual metrics dict
        params: Current parameter values
        weights: Optional custom weights

    Returns:
        Dict with all component scores and total
    """
    if weights is None:
        weights = RewardWeights()

    fps = metrics.get("fps", 60.0)
    entropy = metrics.get("entropy", 5.0)
    color_variance = metrics.get("color_variance", 0.15)
    motion_intensity = metrics.get("motion_intensity", 0.3)

    entropy_score = calculate_entropy_score(entropy)
    color_harmony = calculate_color_harmony_score(color_variance)
    motion_coherence = calculate_motion_coherence_score(motion_intensity)
    fps_penalty = calculate_fps_penalty(fps)
    novelty_bonus = calculate_novelty_bonus(params) if params else 0.5

    total = calculate_total_reward(metrics, params, weights)

    return {
        "entropy_score": entropy_score,
        "color_harmony": color_harmony,
        "motion_coherence": motion_coherence,
        "novelty_bonus": novelty_bonus,
        "fps_penalty": fps_penalty,
        "total_reward": total,
        "weights": {
            "entropy": weights.entropy,
            "color_harmony": weights.color_harmony,
            "motion_coherence": weights.motion_coherence,
            "novelty": weights.novelty,
            "fps_penalty": weights.fps_penalty
        }
    }


def reset_tracking_state():
    """Reset all tracking state (for new sessions)"""
    global _previous_motion, _visited_regions, _total_visits
    _previous_motion = None
    _visited_regions = defaultdict(int)
    _total_visits = 0


def get_exploration_stats() -> Dict[str, Any]:
    """Get current exploration statistics"""
    global _visited_regions, _total_visits

    unique_regions = len(_visited_regions)
    exploration_ratio = unique_regions / _total_visits if _total_visits > 0 else 0.0

    return {
        "unique_regions": unique_regions,
        "total_visits": _total_visits,
        "exploration_ratio": exploration_ratio
    }


# Image analysis functions (requires numpy/PIL)

def compute_image_entropy(image_array: np.ndarray) -> float:
    """
    Compute Shannon entropy of an image.

    Args:
        image_array: Grayscale or RGB image as numpy array

    Returns:
        Entropy value (0-8 for 8-bit images)
    """
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=2).astype(np.uint8)
    else:
        gray = image_array.astype(np.uint8)

    # Compute histogram
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize

    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]

    # Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def compute_color_variance(image_array: np.ndarray) -> float:
    """
    Compute color variance in HSV space.

    Args:
        image_array: RGB image as numpy array

    Returns:
        Variance of hue values (0-1)
    """
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        return 0.0

    # Simple RGB to HSV conversion for hue
    r, g, b = image_array[:,:,0], image_array[:,:,1], image_array[:,:,2]
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c

    # Compute hue
    hue = np.zeros_like(max_c)
    mask = diff != 0

    # Where max is red
    mask_r = mask & (max_c == r)
    hue[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6

    # Where max is green
    mask_g = mask & (max_c == g)
    hue[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2

    # Where max is blue
    mask_b = mask & (max_c == b)
    hue[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4

    hue = hue / 6.0  # Normalize to 0-1

    # Return variance of hue
    return float(np.var(hue))


def compute_motion_intensity(current_frame: np.ndarray,
                             previous_frame: np.ndarray) -> float:
    """
    Compute motion intensity between two frames.

    Simple approach using absolute difference.

    Args:
        current_frame: Current frame as numpy array
        previous_frame: Previous frame as numpy array

    Returns:
        Motion intensity (0-1)
    """
    if current_frame.shape != previous_frame.shape:
        return 0.0

    # Convert to grayscale if needed
    if len(current_frame.shape) == 3:
        current_gray = np.mean(current_frame, axis=2)
        previous_gray = np.mean(previous_frame, axis=2)
    else:
        current_gray = current_frame
        previous_gray = previous_frame

    # Compute absolute difference
    diff = np.abs(current_gray.astype(float) - previous_gray.astype(float))

    # Normalize by max possible difference
    motion = np.mean(diff) / 255.0

    return float(motion)


if __name__ == "__main__":
    # Test the reward functions
    test_metrics = {
        "fps": 45,
        "entropy": 5.5,
        "color_variance": 0.15,
        "motion_intensity": 0.3
    }
    test_params = {
        "shader.zoom": 1.5,
        "shader.speed": 0.8,
        "shader.colorShift": 0.3
    }

    components = get_reward_components(test_metrics, test_params)
    print("Reward Components:")
    for key, value in components.items():
        if key != "weights":
            print(f"  {key}: {value:.3f}")

    print(f"\nTotal Reward: {components['total_reward']:.3f}")
    print(f"\nExploration Stats: {get_exploration_stats()}")
