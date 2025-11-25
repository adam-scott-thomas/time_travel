#!/usr/bin/env python3
"""
Focused experiment with smaller portal widths that will complete faster.
Also includes profiling to identify optimization opportunities.
"""

import json
import time
import cProfile
import pstats
from pathlib import Path
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

from simulation import SimConfig, SimResult, TimeTravelSimulator


def _run_single(args):
    """Worker function."""
    config, max_trips, seed = args
    sim = TimeTravelSimulator(config)
    result = sim.run(max_trips=max_trips, seed=seed)

    return {
        'rule': config.rule,
        'width': config.portal_width,
        'density': config.init_density,
        'seed': seed,
        'found_loop': result.found_loop,
        'cycle_length': result.cycle_length,
        'cycle_start': result.cycle_start,
        'pre_cycle_length': result.pre_cycle_length,
        'total_trips': result.total_trips,
        'is_perfect': result.is_perfect_loop,
    }


def profile_simulation():
    """Profile a single simulation to identify bottlenecks."""
    print("=== Profiling Single Simulation ===\n")

    config = SimConfig(
        rule=110,
        portal_width=16,
        init_density=0.5,
        t_enter=80,
        t_exit=40,
        num_cells=256,
        num_generations=144,
    )

    profiler = cProfile.Profile()
    profiler.enable()

    # Run many simulations
    for seed in range(100):
        sim = TimeTravelSimulator(config)
        result = sim.run(max_trips=10000, seed=seed)

    profiler.disable()

    print("Top 20 functions by cumulative time:")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    print("\nTop 20 functions by total time:")
    stats.sort_stats('tottime')
    stats.print_stats(20)

    # Save profile for detailed analysis
    stats.dump_stats('simulation.prof')
    print("\nProfile saved to simulation.prof")


def run_focused_experiment():
    """Run focused experiment with smaller portal widths."""
    print("=== Focused Time Travel Experiment ===\n")

    # Selected chaotic rules
    rules = [30, 45, 73, 90, 110, 150, 169, 182]

    # Smaller portal widths - these complete in reasonable time
    # State space grows as 2^width, so we cap at 20 cells (1M states)
    widths = list(range(6, 22, 2))  # 6, 8, 10, 12, 14, 16, 18, 20

    # Initial densities
    densities = [0.1, 0.3, 0.5, 0.7, 0.9]

    reps = 500  # More reps for better statistics
    max_trips = 100000  # Allow longer runs

    total = len(rules) * len(widths) * len(densities) * reps
    print(f"Configuration:")
    print(f"  Rules: {rules}")
    print(f"  Widths: {widths}")
    print(f"  Densities: {densities}")
    print(f"  Reps per config: {reps}")
    print(f"  Total experiments: {total:,}")
    print()

    configs = []
    for rule in rules:
        for width in widths:
            for density in densities:
                for rep in range(reps):
                    config = SimConfig(
                        rule=rule,
                        portal_width=width,
                        init_density=density,
                    )
                    seed = hash((rule, width, density, rep)) % (2**31)
                    configs.append((config, max_trips, seed))

    results = []
    print(f"Running experiments with {cpu_count()} workers...")
    start = time.time()

    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap(_run_single, configs, chunksize=100), total=total):
            results.append(res)

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s ({total/elapsed:.1f} experiments/sec)")

    # Aggregate and save results
    save_results(results)

    return results


def save_results(results):
    """Aggregate and save experiment results."""
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Overall statistics
    overall = {
        'total': len(results),
        'loops_found': sum(1 for r in results if r['found_loop']),
        'perfect_loops': sum(1 for r in results if r['is_perfect']),
        'oscillating_loops': sum(1 for r in results if r['found_loop'] and not r['is_perfect']),
    }
    overall['loop_rate'] = overall['loops_found'] / overall['total']
    overall['perfect_rate'] = overall['perfect_loops'] / overall['total']
    overall['oscillating_rate'] = overall['oscillating_loops'] / overall['total']

    cycle_lengths = [r['cycle_length'] for r in results if r['found_loop']]
    pre_cycles = [r['pre_cycle_length'] for r in results if r['found_loop']]

    overall['mean_cycle_length'] = float(np.mean(cycle_lengths)) if cycle_lengths else 0
    overall['max_cycle_length'] = int(max(cycle_lengths)) if cycle_lengths else 0
    overall['mean_pre_cycle'] = float(np.mean(pre_cycles)) if pre_cycles else 0
    overall['max_pre_cycle'] = int(max(pre_cycles)) if pre_cycles else 0

    # By rule
    by_rule = defaultdict(list)
    for r in results:
        by_rule[r['rule']].append(r)

    rule_summary = {}
    for rule, data in by_rule.items():
        cycles = [d['cycle_length'] for d in data if d['found_loop']]
        pre = [d['pre_cycle_length'] for d in data if d['found_loop']]
        perfect = sum(1 for d in data if d['is_perfect'])
        oscillating = sum(1 for d in data if d['found_loop'] and not d['is_perfect'])

        rule_summary[int(rule)] = {
            'total': len(data),
            'loops_found': sum(1 for d in data if d['found_loop']),
            'perfect_loops': perfect,
            'oscillating_loops': oscillating,
            'loop_rate': sum(1 for d in data if d['found_loop']) / len(data),
            'perfect_rate': perfect / len(data),
            'oscillating_rate': oscillating / len(data),
            'mean_cycle_length': float(np.mean(cycles)) if cycles else 0,
            'max_cycle_length': int(max(cycles)) if cycles else 0,
            'mean_pre_cycle': float(np.mean(pre)) if pre else 0,
            'max_pre_cycle': int(max(pre)) if pre else 0,
            'cycle_length_dist': {int(k): v for k, v in Counter(cycles).items()},
        }

    # By width
    by_width = defaultdict(list)
    for r in results:
        by_width[r['width']].append(r)

    width_summary = {}
    for width, data in by_width.items():
        cycles = [d['cycle_length'] for d in data if d['found_loop']]
        pre = [d['pre_cycle_length'] for d in data if d['found_loop']]
        perfect = sum(1 for d in data if d['is_perfect'])
        oscillating = sum(1 for d in data if d['found_loop'] and not d['is_perfect'])

        state_space = 2**width
        expected_random = np.sqrt(np.pi * state_space / 2)  # Birthday paradox

        width_summary[int(width)] = {
            'total': len(data),
            'state_space': state_space,
            'expected_random_trips': float(expected_random),
            'loops_found': sum(1 for d in data if d['found_loop']),
            'perfect_loops': perfect,
            'oscillating_loops': oscillating,
            'loop_rate': sum(1 for d in data if d['found_loop']) / len(data),
            'perfect_rate': perfect / len(data),
            'oscillating_rate': oscillating / len(data),
            'mean_cycle_length': float(np.mean(cycles)) if cycles else 0,
            'max_cycle_length': int(max(cycles)) if cycles else 0,
            'mean_pre_cycle': float(np.mean(pre)) if pre else 0,
            'max_pre_cycle': int(max(pre)) if pre else 0,
        }

    # By density
    by_density = defaultdict(list)
    for r in results:
        by_density[r['density']].append(r)

    density_summary = {}
    for density, data in by_density.items():
        cycles = [d['cycle_length'] for d in data if d['found_loop']]
        pre = [d['pre_cycle_length'] for d in data if d['found_loop']]
        perfect = sum(1 for d in data if d['is_perfect'])
        oscillating = sum(1 for d in data if d['found_loop'] and not d['is_perfect'])

        density_summary[float(density)] = {
            'total': len(data),
            'loops_found': sum(1 for d in data if d['found_loop']),
            'perfect_loops': perfect,
            'oscillating_loops': oscillating,
            'loop_rate': sum(1 for d in data if d['found_loop']) / len(data),
            'perfect_rate': perfect / len(data),
            'mean_cycle_length': float(np.mean(cycles)) if cycles else 0,
            'mean_pre_cycle': float(np.mean(pre)) if pre else 0,
        }

    all_data = {
        'overall': overall,
        'by_rule': rule_summary,
        'by_width': width_summary,
        'by_density': density_summary,
    }

    with open(output_dir / 'focused_results.json', 'w') as f:
        json.dump(all_data, f, indent=2)

    print("\nSaved to results/focused_results.json")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nOverall: {overall['total']:,} experiments")
    print(f"  Loops found: {overall['loop_rate']:.2%}")
    print(f"  Perfect loops: {overall['perfect_rate']:.2%}")
    print(f"  Oscillating loops: {overall['oscillating_rate']:.2%}")
    print(f"  Mean cycle length: {overall['mean_cycle_length']:.2f}")
    print(f"  Max cycle length: {overall['max_cycle_length']}")
    print(f"  Mean pre-cycle: {overall['mean_pre_cycle']:.2f}")
    print(f"  Max pre-cycle: {overall['max_pre_cycle']}")

    print("\nBy Rule:")
    for rule in sorted(rule_summary.keys()):
        s = rule_summary[rule]
        print(f"  Rule {rule:3d}: loop={s['loop_rate']:.0%}, "
              f"perfect={s['perfect_rate']:.1%}, oscillating={s['oscillating_rate']:.1%}, "
              f"mean_cycle={s['mean_cycle_length']:.1f}, max_cycle={s['max_cycle_length']}")

    print("\nBy Width:")
    for width in sorted(width_summary.keys()):
        s = width_summary[width]
        print(f"  Width {width:2d} (2^{width}={s['state_space']:>8,}): loop={s['loop_rate']:.0%}, "
              f"mean_cycle={s['mean_cycle_length']:.1f}, mean_pre={s['mean_pre_cycle']:.1f} "
              f"(expected: {s['expected_random_trips']:.0f})")

    print("\nBy Density:")
    for density in sorted(density_summary.keys()):
        s = density_summary[density]
        print(f"  Density {density:.1f}: loop={s['loop_rate']:.0%}, "
              f"perfect={s['perfect_rate']:.1%}, mean_cycle={s['mean_cycle_length']:.1f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        profile_simulation()
    else:
        run_focused_experiment()
