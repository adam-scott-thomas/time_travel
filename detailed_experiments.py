#!/usr/bin/env python3
"""
More detailed experiments to investigate:
1. Do oscillating loops exist at all? (need very long runs)
2. How does behavior vary across different CA rules?
3. What's the relationship between portal width and time to loop?
"""

import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from tqdm import tqdm

from simulation import SimConfig, SimResult, TimeTravelSimulator


def _run_single_detailed(args):
    """Worker that returns detailed information."""
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


def investigate_oscillating_loops():
    """Try hard to find oscillating loops (cycle_length > 1)."""
    print("=== Hunting for Oscillating Loops ===\n")

    # Test many different rules with long runs
    rules = list(range(256))  # All 256 Wolfram rules
    widths = [16, 20, 24, 28, 32]
    seeds_per_config = 50
    max_trips = 10000  # Very long runs

    oscillating_found = []

    for rule in tqdm(rules, desc="Testing rules"):
        for width in widths:
            for seed in range(seeds_per_config):
                config = SimConfig(
                    rule=rule,
                    portal_width=width,
                    init_density=0.5,
                )
                sim = TimeTravelSimulator(config)
                result = sim.run(max_trips=max_trips, seed=seed)

                if result.found_loop and result.cycle_length > 1:
                    oscillating_found.append({
                        'rule': rule,
                        'width': width,
                        'seed': seed,
                        'cycle_length': result.cycle_length,
                        'pre_cycle': result.pre_cycle_length,
                    })

    print(f"\nFound {len(oscillating_found)} oscillating loops!")
    if oscillating_found:
        print("\nExamples:")
        for item in oscillating_found[:20]:
            print(f"  Rule {item['rule']}, width {item['width']}: "
                  f"cycle_length={item['cycle_length']}, pre_cycle={item['pre_cycle']}")

    return oscillating_found


def analyze_convergence_speed():
    """Analyze how quickly loops are found for different configurations."""
    print("=== Analyzing Convergence Speed ===\n")

    rules = [30, 45, 73, 90, 97, 110, 137, 150, 161, 165, 169, 182]
    widths = list(range(8, 40, 2))
    reps = 500
    max_trips = 50000

    results = defaultdict(list)

    total = len(rules) * len(widths) * reps
    configs = []
    for rule in rules:
        for width in widths:
            for rep in range(reps):
                config = SimConfig(rule=rule, portal_width=width, init_density=0.5)
                seed = hash((rule, width, rep)) % (2**31)
                configs.append((config, max_trips, seed))

    print(f"Running {total:,} experiments...")
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap(_run_single_detailed, configs, chunksize=100), total=total):
            key = (res['rule'], res['width'])
            results[key].append(res)

    # Aggregate results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    summary = {}
    for (rule, width), data in results.items():
        loops_found = sum(1 for d in data if d['found_loop'])
        perfect_loops = sum(1 for d in data if d['is_perfect'])
        oscillating = sum(1 for d in data if d['found_loop'] and not d['is_perfect'])

        cycle_lengths = [d['cycle_length'] for d in data if d['found_loop']]
        pre_cycles = [d['pre_cycle_length'] for d in data if d['found_loop']]
        total_trips = [d['total_trips'] for d in data]

        summary[f"{rule}_{width}"] = {
            'rule': rule,
            'width': width,
            'total': len(data),
            'loops_found': loops_found,
            'perfect_loops': perfect_loops,
            'oscillating_loops': oscillating,
            'loop_rate': loops_found / len(data),
            'perfect_rate': perfect_loops / len(data),
            'oscillating_rate': oscillating / len(data),
            'mean_cycle_length': np.mean(cycle_lengths) if cycle_lengths else 0,
            'max_cycle_length': max(cycle_lengths) if cycle_lengths else 0,
            'mean_pre_cycle': np.mean(pre_cycles) if pre_cycles else 0,
            'max_pre_cycle': max(pre_cycles) if pre_cycles else 0,
            'mean_total_trips': np.mean(total_trips),
            'cycle_length_dist': dict(Counter(cycle_lengths)),
        }

    with open(output_dir / 'convergence_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to results/convergence_analysis.json")

    # Print summary
    print("\n=== Summary by Rule ===")
    by_rule = defaultdict(list)
    for key, data in summary.items():
        by_rule[data['rule']].append(data)

    for rule in sorted(by_rule.keys()):
        rule_data = by_rule[rule]
        avg_loop_rate = np.mean([d['loop_rate'] for d in rule_data])
        avg_perfect_rate = np.mean([d['perfect_rate'] for d in rule_data])
        avg_oscillating = np.mean([d['oscillating_rate'] for d in rule_data])
        avg_cycle = np.mean([d['mean_cycle_length'] for d in rule_data])

        print(f"Rule {rule:3d}: loop_rate={avg_loop_rate:.2%}, "
              f"perfect={avg_perfect_rate:.2%}, oscillating={avg_oscillating:.2%}, "
              f"mean_cycle={avg_cycle:.2f}")

    return summary


def analyze_pre_cycle_distribution():
    """Deep dive into pre-cycle lengths - how long before loops start?"""
    print("=== Analyzing Pre-Cycle Distribution ===\n")

    # Focus on one chaotic rule with varying widths
    rule = 110
    widths = list(range(6, 36, 2))
    reps = 2000
    max_trips = 100000

    all_results = []

    total = len(widths) * reps
    configs = []
    for width in widths:
        for rep in range(reps):
            config = SimConfig(rule=rule, portal_width=width, init_density=0.5)
            seed = hash((rule, width, rep)) % (2**31)
            configs.append((config, max_trips, seed))

    print(f"Running {total:,} experiments for Rule {rule}...")
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap(_run_single_detailed, configs, chunksize=100), total=total):
            all_results.append(res)

    # Group by width
    by_width = defaultdict(list)
    for res in all_results:
        by_width[res['width']].append(res)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    analysis = {}
    print("\n=== Pre-Cycle Analysis by Width ===")
    for width in sorted(by_width.keys()):
        data = by_width[width]
        pre_cycles = [d['pre_cycle_length'] for d in data if d['found_loop']]
        total_trips_list = [d['total_trips'] for d in data if d['found_loop']]

        analysis[width] = {
            'width': width,
            'state_space': 2**width,
            'num_samples': len(pre_cycles),
            'mean_pre_cycle': np.mean(pre_cycles) if pre_cycles else 0,
            'median_pre_cycle': np.median(pre_cycles) if pre_cycles else 0,
            'std_pre_cycle': np.std(pre_cycles) if pre_cycles else 0,
            'max_pre_cycle': max(pre_cycles) if pre_cycles else 0,
            'mean_total_trips': np.mean(total_trips_list) if total_trips_list else 0,
            'pre_cycle_dist': dict(Counter(pre_cycles)),
        }

        state_space = 2**width
        expected_random = np.sqrt(np.pi * state_space / 2)  # Birthday paradox estimate

        print(f"Width {width:2d} (2^{width}={state_space:>12,}): "
              f"mean_pre_cycle={analysis[width]['mean_pre_cycle']:.1f}, "
              f"expected_random={expected_random:.1f}")

    with open(output_dir / 'pre_cycle_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print("\nSaved to results/pre_cycle_analysis.json")
    return analysis


def run_comprehensive_experiment():
    """Run the main comprehensive experiment."""
    print("=== Comprehensive Time Travel Experiment ===\n")

    # Good chaotic rules from prior analysis
    rules = [30, 45, 73, 90, 97, 110, 137, 150, 161, 165, 169, 182]

    # Wide range of portal widths
    widths = list(range(6, 32, 2))

    # Various initial densities
    densities = [0.1, 0.3, 0.5, 0.7, 0.9]

    reps = 300
    max_trips = 50000

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
        for res in tqdm(pool.imap(_run_single_detailed, configs, chunksize=100), total=total):
            results.append(res)

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s ({total/elapsed:.1f} experiments/sec)")

    # Aggregate results
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

    overall['mean_cycle_length'] = np.mean(cycle_lengths) if cycle_lengths else 0
    overall['mean_pre_cycle'] = np.mean(pre_cycles) if pre_cycles else 0
    overall['cycle_length_dist'] = dict(Counter(cycle_lengths))
    overall['pre_cycle_dist'] = dict(Counter(pre_cycles))

    # By rule
    by_rule = defaultdict(list)
    for r in results:
        by_rule[r['rule']].append(r)

    rule_summary = {}
    for rule, data in by_rule.items():
        cycles = [d['cycle_length'] for d in data if d['found_loop']]
        pre = [d['pre_cycle_length'] for d in data if d['found_loop']]
        rule_summary[rule] = {
            'total': len(data),
            'loops_found': sum(1 for d in data if d['found_loop']),
            'perfect_loops': sum(1 for d in data if d['is_perfect']),
            'oscillating_loops': sum(1 for d in data if d['found_loop'] and not d['is_perfect']),
            'loop_rate': sum(1 for d in data if d['found_loop']) / len(data),
            'mean_cycle_length': np.mean(cycles) if cycles else 0,
            'mean_pre_cycle': np.mean(pre) if pre else 0,
            'cycle_length_dist': dict(Counter(cycles)),
        }

    # By width
    by_width = defaultdict(list)
    for r in results:
        by_width[r['width']].append(r)

    width_summary = {}
    for width, data in by_width.items():
        cycles = [d['cycle_length'] for d in data if d['found_loop']]
        pre = [d['pre_cycle_length'] for d in data if d['found_loop']]
        width_summary[width] = {
            'total': len(data),
            'loops_found': sum(1 for d in data if d['found_loop']),
            'perfect_loops': sum(1 for d in data if d['is_perfect']),
            'oscillating_loops': sum(1 for d in data if d['found_loop'] and not d['is_perfect']),
            'loop_rate': sum(1 for d in data if d['found_loop']) / len(data),
            'mean_cycle_length': np.mean(cycles) if cycles else 0,
            'mean_pre_cycle': np.mean(pre) if pre else 0,
        }

    # By density
    by_density = defaultdict(list)
    for r in results:
        by_density[r['density']].append(r)

    density_summary = {}
    for density, data in by_density.items():
        cycles = [d['cycle_length'] for d in data if d['found_loop']]
        pre = [d['pre_cycle_length'] for d in data if d['found_loop']]
        density_summary[density] = {
            'total': len(data),
            'loops_found': sum(1 for d in data if d['found_loop']),
            'perfect_loops': sum(1 for d in data if d['is_perfect']),
            'loop_rate': sum(1 for d in data if d['found_loop']) / len(data),
            'mean_cycle_length': np.mean(cycles) if cycles else 0,
            'mean_pre_cycle': np.mean(pre) if pre else 0,
        }

    all_data = {
        'overall': overall,
        'by_rule': rule_summary,
        'by_width': width_summary,
        'by_density': density_summary,
    }

    with open(output_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(all_data, f, indent=2)

    print("\nSaved to results/comprehensive_results.json")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nOverall: {overall['total']:,} experiments")
    print(f"  Loops found: {overall['loop_rate']:.2%}")
    print(f"  Perfect loops: {overall['perfect_rate']:.2%}")
    print(f"  Oscillating loops: {overall['oscillating_rate']:.2%}")
    print(f"  Mean cycle length: {overall['mean_cycle_length']:.2f}")
    print(f"  Mean pre-cycle: {overall['mean_pre_cycle']:.2f}")

    print("\nBy Rule:")
    for rule in sorted(rule_summary.keys()):
        s = rule_summary[rule]
        print(f"  Rule {rule:3d}: loop={s['loop_rate']:.2%}, "
              f"perfect={s['perfect_loops']/s['total']:.2%}, "
              f"mean_cycle={s['mean_cycle_length']:.1f}")

    print("\nBy Width:")
    for width in sorted(width_summary.keys()):
        s = width_summary[width]
        print(f"  Width {width:2d}: loop={s['loop_rate']:.2%}, "
              f"mean_cycle={s['mean_cycle_length']:.1f}, "
              f"mean_pre_cycle={s['mean_pre_cycle']:.1f}")

    return all_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "oscillating":
            investigate_oscillating_loops()
        elif sys.argv[1] == "convergence":
            analyze_convergence_speed()
        elif sys.argv[1] == "precycle":
            analyze_pre_cycle_distribution()
        elif sys.argv[1] == "full":
            run_comprehensive_experiment()
    else:
        print("Usage: python detailed_experiments.py [oscillating|convergence|precycle|full]")
        print("  oscillating - Hunt for oscillating loops (cycle_length > 1)")
        print("  convergence - Analyze convergence speed across rules")
        print("  precycle    - Analyze pre-cycle length distribution")
        print("  full        - Run comprehensive experiment")
