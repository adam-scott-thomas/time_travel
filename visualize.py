#!/usr/bin/env python3
"""
Visualization module for time travel paradox analysis.

Generates:
1. Statistical plots showing distributions and trends
2. Animations of cellular automata with time travel
3. Figures for the blog post
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from collections import Counter

from simulation import SimConfig, TimeTravelSimulator


def load_results(results_dir: str = "results", name: str = "full_experiment"):
    """Load experiment results from disk."""
    results_dir = Path(results_dir)

    # Load aggregated results
    json_path = results_dir / f"{name}_aggregated.json"
    with open(json_path, 'r') as f:
        aggregated = json.load(f)

    return aggregated


def plot_width_scaling(results: Dict, output_dir: str = "figures"):
    """Plot how loop properties scale with portal width."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    by_width = results.get('by_width', {})
    if not by_width:
        print("No width data found")
        return

    widths = sorted([int(w) for w in by_width.keys()])
    loop_rates = [by_width[str(w)]['loop_rate'] for w in widths]
    perfect_rates = [by_width[str(w)]['perfect_loop_rate'] for w in widths]
    mean_cycles = [by_width[str(w)]['mean_cycle_length'] for w in widths]
    mean_pre_cycles = [by_width[str(w)]['mean_pre_cycle_length'] for w in widths]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Loop rate vs width
    ax = axes[0, 0]
    ax.plot(widths, loop_rates, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Portal Width (cells)', fontsize=12)
    ax.set_ylabel('Loop Detection Rate', fontsize=12)
    ax.set_title('Loop Detection Rate vs Portal Width', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 2. Perfect vs oscillating loop rates
    ax = axes[0, 1]
    oscillating_rates = [loop_rates[i] - perfect_rates[i] for i in range(len(widths))]
    ax.bar(widths, perfect_rates, label='Perfect Loops', alpha=0.7, color='green')
    ax.bar(widths, oscillating_rates, bottom=perfect_rates, label='Oscillating Loops', alpha=0.7, color='orange')
    ax.set_xlabel('Portal Width (cells)', fontsize=12)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_title('Perfect vs Oscillating Loops', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Mean cycle length vs width (log scale)
    ax = axes[1, 0]
    ax.semilogy(widths, mean_cycles, 'r-o', linewidth=2, markersize=6)
    ax.set_xlabel('Portal Width (cells)', fontsize=12)
    ax.set_ylabel('Mean Cycle Length (log scale)', fontsize=12)
    ax.set_title('Mean Cycle Length vs Portal Width', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add theoretical 2^width line for comparison
    theoretical = [2**(w/2) for w in widths]  # sqrt of state space
    ax.semilogy(widths, theoretical, 'k--', alpha=0.5, label='$2^{width/2}$')
    ax.legend()

    # 4. Mean pre-cycle length vs width (log scale)
    ax = axes[1, 1]
    ax.semilogy(widths, mean_pre_cycles, 'g-o', linewidth=2, markersize=6)
    ax.set_xlabel('Portal Width (cells)', fontsize=12)
    ax.set_ylabel('Mean Pre-cycle Length (log scale)', fontsize=12)
    ax.set_title('Mean Iterations Before Loop vs Portal Width', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'width_scaling.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved width_scaling.png")


def plot_cycle_distribution(results: Dict, output_dir: str = "figures"):
    """Plot the distribution of cycle lengths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    overall = results.get('overall', {})
    cycle_lengths = overall.get('cycle_lengths', [])
    pre_cycle_lengths = overall.get('pre_cycle_lengths', [])

    if not cycle_lengths:
        print("No cycle data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Cycle length distribution (histogram)
    ax = axes[0]
    cycle_counts = Counter(cycle_lengths)
    max_cycle = min(50, max(cycle_lengths))  # Cap for readability
    xs = list(range(1, max_cycle + 1))
    ys = [cycle_counts.get(x, 0) for x in xs]

    ax.bar(xs, ys, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Cycle Length (number of phases)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Cycle Lengths', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight perfect loops (cycle_length = 1)
    if 1 in cycle_counts:
        ax.bar([1], [cycle_counts[1]], color='green', alpha=0.9, edgecolor='black', label=f'Perfect loops: {cycle_counts[1]}')
        ax.legend()

    # 2. Pre-cycle length distribution
    ax = axes[1]
    pre_cycle_counts = Counter(pre_cycle_lengths)
    max_pre = min(100, max(pre_cycle_lengths) if pre_cycle_lengths else 1)
    xs = list(range(0, max_pre + 1))
    ys = [pre_cycle_counts.get(x, 0) for x in xs]

    ax.bar(xs, ys, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Pre-cycle Length (iterations before loop)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Iterations Before Loop', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'cycle_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved cycle_distributions.png")


def plot_rule_comparison(results: Dict, output_dir: str = "figures"):
    """Compare behavior across different CA rules."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    by_rule = results.get('by_rule', {})
    if not by_rule:
        print("No rule data found")
        return

    rules = sorted([int(r) for r in by_rule.keys()])
    loop_rates = [by_rule[str(r)]['loop_rate'] for r in rules]
    perfect_rates = [by_rule[str(r)]['perfect_loop_rate'] for r in rules]
    mean_cycles = [by_rule[str(r)]['mean_cycle_length'] for r in rules]
    mean_pre_cycles = [by_rule[str(r)]['mean_pre_cycle_length'] for r in rules]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loop rates by rule
    ax = axes[0, 0]
    x_pos = np.arange(len(rules))
    ax.bar(x_pos, loop_rates, color='steelblue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in rules])
    ax.set_xlabel('Wolfram Rule Number', fontsize=12)
    ax.set_ylabel('Loop Detection Rate', fontsize=12)
    ax.set_title('Loop Rate by CA Rule', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Perfect loop rates
    ax = axes[0, 1]
    ax.bar(x_pos, perfect_rates, color='green', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in rules])
    ax.set_xlabel('Wolfram Rule Number', fontsize=12)
    ax.set_ylabel('Perfect Loop Rate', fontsize=12)
    ax.set_title('Perfect Loop Rate by CA Rule', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Mean cycle length by rule
    ax = axes[1, 0]
    ax.bar(x_pos, mean_cycles, color='coral', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in rules])
    ax.set_xlabel('Wolfram Rule Number', fontsize=12)
    ax.set_ylabel('Mean Cycle Length', fontsize=12)
    ax.set_title('Mean Cycle Length by CA Rule', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Mean pre-cycle length by rule
    ax = axes[1, 1]
    ax.bar(x_pos, mean_pre_cycles, color='purple', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(r) for r in rules])
    ax.set_xlabel('Wolfram Rule Number', fontsize=12)
    ax.set_ylabel('Mean Pre-cycle Length', fontsize=12)
    ax.set_title('Mean Iterations Before Loop by CA Rule', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'rule_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rule_comparison.png")


def plot_density_effect(results: Dict, output_dir: str = "figures"):
    """Plot how initial density affects loop behavior."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    by_density = results.get('by_density', {})
    if not by_density:
        print("No density data found")
        return

    densities = sorted([float(d) for d in by_density.keys()])
    loop_rates = [by_density[str(d)]['loop_rate'] for d in densities]
    perfect_rates = [by_density[str(d)]['perfect_loop_rate'] for d in densities]
    mean_cycles = [by_density[str(d)]['mean_cycle_length'] for d in densities]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(densities, loop_rates, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Initial Density', fontsize=12)
    ax.set_ylabel('Loop Detection Rate', fontsize=12)
    ax.set_title('Loop Rate vs Initial Density', fontsize=14)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(densities, perfect_rates, 'g-o', linewidth=2, markersize=8)
    ax.set_xlabel('Initial Density', fontsize=12)
    ax.set_ylabel('Perfect Loop Rate', fontsize=12)
    ax.set_title('Perfect Loop Rate vs Initial Density', fontsize=14)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(densities, mean_cycles, 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('Initial Density', fontsize=12)
    ax.set_ylabel('Mean Cycle Length', fontsize=12)
    ax.set_title('Mean Cycle Length vs Initial Density', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'density_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved density_effect.png")


def create_ca_animation(
    config: SimConfig,
    output_path: str = "figures/ca_animation.gif",
    max_frames: int = 200,
    fps: int = 10,
    seed: int = 42
):
    """Create an animated GIF of cellular automata with time travel."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    sim = TimeTravelSimulator(config)
    sim.reset(seed=seed)

    # Collect frames
    frames = []
    portal_events = []  # Track when time travel happens

    for frame_num in range(max_frames):
        # Record state before step
        prev_gen = sim.current_gen

        # Take step
        loop_found = sim.step()

        # Check for time travel (generation went backwards)
        if sim.current_gen < prev_gen:
            portal_events.append(frame_num)

        # Store universe snapshot
        frames.append(sim.universe.copy())

        if loop_found:
            # Add a few more frames to show the loop
            for _ in range(20):
                sim.step()
                frames.append(sim.universe.copy())
            break

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))

    # Custom colormap: black for 0, white for 1
    cmap = ListedColormap(['black', 'white'])

    # Initial plot
    im = ax.imshow(frames[0], cmap=cmap, aspect='auto', interpolation='nearest')

    # Portal rectangles
    cfg = config
    portal_x = cfg.num_cells // 2 + cfg.portal_offset
    exit_rect = Rectangle((portal_x - 0.5, cfg.t_exit - 0.5), cfg.portal_width, 1,
                           linewidth=2, edgecolor='red', facecolor='none', label='Exit')
    enter_rect = Rectangle((portal_x - 0.5, cfg.t_enter - 0.5), cfg.portal_width, 1,
                            linewidth=2, edgecolor='green', facecolor='none', label='Entry')
    ax.add_patch(exit_rect)
    ax.add_patch(enter_rect)

    ax.set_xlabel('Cell Position')
    ax.set_ylabel('Time Step')
    title = ax.set_title(f'Rule {cfg.rule}, Width {cfg.portal_width} - Frame 0')
    ax.legend(loc='upper right')

    def update(frame_idx):
        im.set_array(frames[frame_idx])

        # Update title
        status = ""
        if frame_idx in portal_events:
            status = " - TIME TRAVEL!"
        if sim.result and frame_idx >= len(frames) - 20:
            status = f" - LOOP FOUND! (cycle={sim.result.cycle_length})"
        title.set_text(f'Rule {cfg.rule}, Width {cfg.portal_width} - Frame {frame_idx}{status}')

        return [im, title]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=False)

    # Save as GIF
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved animation to {output_path}")

    return sim.result


def create_static_ca_figure(
    config: SimConfig,
    output_path: str = "figures/ca_example.png",
    seed: int = 42
):
    """Create a static figure showing CA evolution with time travel."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    sim = TimeTravelSimulator(config)
    result = sim.run(max_trips=100, seed=seed)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Custom colormap
    cmap = ListedColormap(['black', 'white'])

    im = ax.imshow(sim.universe, cmap=cmap, aspect='auto', interpolation='nearest')

    # Portal rectangles
    cfg = config
    portal_x = cfg.num_cells // 2 + cfg.portal_offset
    exit_rect = Rectangle((portal_x - 0.5, cfg.t_exit - 0.5), cfg.portal_width, 2,
                           linewidth=3, edgecolor='red', facecolor='red', alpha=0.3, label='Time Exit')
    enter_rect = Rectangle((portal_x - 0.5, cfg.t_enter - 0.5), cfg.portal_width, 2,
                            linewidth=3, edgecolor='green', facecolor='green', alpha=0.3, label='Time Entry')
    ax.add_patch(exit_rect)
    ax.add_patch(enter_rect)

    # Draw arrow showing time travel direction
    ax.annotate('', xy=(portal_x + cfg.portal_width/2, cfg.t_exit + 5),
                xytext=(portal_x + cfg.portal_width/2, cfg.t_enter - 5),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=2))
    ax.text(portal_x + cfg.portal_width + 5, (cfg.t_enter + cfg.t_exit)/2,
            'Time Travel', fontsize=10, color='cyan', va='center')

    ax.set_xlabel('Cell Position', fontsize=12)
    ax.set_ylabel('Time Step', fontsize=12)
    ax.set_title(f'1D Cellular Automata with Time Travel (Rule {cfg.rule})', fontsize=14)
    ax.legend(loc='upper right')

    # Add result info
    if result.found_loop:
        info_text = f"Loop found!\nCycle length: {result.cycle_length}\nPre-cycle: {result.pre_cycle_length}"
    else:
        info_text = "No loop found"
    ax.text(0.02, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved static figure to {output_path}")

    return result


def create_loop_examples(output_dir: str = "figures"):
    """Create example figures showing different types of loops."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find examples of different loop types
    configs_to_try = [
        SimConfig(rule=110, portal_width=10, init_density=0.5),
        SimConfig(rule=30, portal_width=12, init_density=0.5),
        SimConfig(rule=45, portal_width=8, init_density=0.3),
    ]

    for i, cfg in enumerate(configs_to_try):
        # Try different seeds to find interesting examples
        for seed in range(100):
            sim = TimeTravelSimulator(cfg)
            result = sim.run(max_trips=500, seed=seed)

            if result.found_loop and result.cycle_length <= 10:
                # Found a good example
                create_static_ca_figure(cfg, output_dir / f"ca_example_{i}_seed{seed}.png", seed)
                print(f"  Rule {cfg.rule}: cycle_length={result.cycle_length}, pre_cycle={result.pre_cycle_length}")
                break


def generate_all_figures(results_dir: str = "results", output_dir: str = "figures"):
    """Generate all figures for the blog post."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Try to load experiment results
    try:
        results = load_results(results_dir, "full_experiment")
        has_full = True
    except FileNotFoundError:
        print("Full experiment results not found, trying width_scaling...")
        try:
            results = load_results(results_dir, "width_scaling")
            has_full = False
        except FileNotFoundError:
            print("No experiment results found. Run experiments first.")
            results = None
            has_full = False

    # Generate statistical plots if we have results
    if results:
        print("\n=== Generating Statistical Plots ===")
        plot_width_scaling(results, output_dir)
        plot_cycle_distribution(results, output_dir)
        if has_full:
            plot_rule_comparison(results, output_dir)
            plot_density_effect(results, output_dir)

    # Generate example animations
    print("\n=== Generating Animations ===")

    # Animation 1: Basic time travel visualization
    cfg = SimConfig(rule=110, portal_width=12, init_density=0.5, num_cells=128, num_generations=100)
    create_ca_animation(cfg, output_dir / "time_travel_basic.gif", max_frames=150, fps=8, seed=42)

    # Animation 2: Finding a loop
    cfg = SimConfig(rule=30, portal_width=10, init_density=0.5, num_cells=128, num_generations=100)
    create_ca_animation(cfg, output_dir / "time_travel_loop.gif", max_frames=200, fps=10, seed=123)

    # Static examples
    print("\n=== Generating Static Figures ===")
    create_loop_examples(output_dir)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            generate_all_figures()
        elif sys.argv[1] == "plots":
            results = load_results()
            plot_width_scaling(results)
            plot_cycle_distribution(results)
            plot_rule_comparison(results)
            plot_density_effect(results)
        elif sys.argv[1] == "anim":
            cfg = SimConfig(rule=110, portal_width=12, init_density=0.5)
            create_ca_animation(cfg)
    else:
        print("Usage: python visualize.py [all|plots|anim]")
        print("  all   - Generate all figures and animations")
        print("  plots - Generate statistical plots only")
        print("  anim  - Generate a single animation")
