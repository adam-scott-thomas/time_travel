#!/usr/bin/env python3
"""
Investigate whether the perfect loop behavior is due to boundary conditions.

Hypothesis: The cells outside the portal region are unchanged between trips,
which might be dominating the evolution and forcing convergence to fixed points.

Tests:
1. Vary the ratio of portal_width to time_gap (t_enter - t_exit)
2. Try very wide portals that span most of the universe
3. Try very short time gaps
"""

import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from simulation import SimConfig, TimeTravelSimulator


def experiment_varying_time_gap():
    """Vary the time gap between portal exit and entry."""
    print("=== Experiment 1: Varying Time Gap ===\n")

    results = {}
    portal_width = 32
    num_cells = 256
    num_generations = 200

    for time_gap in [5, 10, 20, 40, 60, 80, 100]:
        t_exit = 20
        t_enter = t_exit + time_gap

        if t_enter >= num_generations - 5:
            continue

        cycles = []
        pre_cycles = []

        for seed in range(500):
            config = SimConfig(
                rule=110,
                portal_width=portal_width,
                init_density=0.5,
                t_enter=t_enter,
                t_exit=t_exit,
                num_cells=num_cells,
                num_generations=num_generations,
            )
            sim = TimeTravelSimulator(config)
            result = sim.run(max_trips=10000, seed=seed)

            if result.found_loop:
                cycles.append(result.cycle_length)
                pre_cycles.append(result.pre_cycle_length)

        results[time_gap] = {
            'mean_cycle': np.mean(cycles) if cycles else 0,
            'max_cycle': max(cycles) if cycles else 0,
            'mean_pre_cycle': np.mean(pre_cycles) if pre_cycles else 0,
            'oscillating': sum(1 for c in cycles if c > 1),
            'cycle_dist': dict(Counter(cycles)),
        }

        print(f"Time gap {time_gap:3d}: mean_cycle={results[time_gap]['mean_cycle']:.1f}, "
              f"oscillating={results[time_gap]['oscillating']}, "
              f"cycle_dist={results[time_gap]['cycle_dist']}")

    return results


def experiment_varying_portal_width():
    """Try very wide portals that approach full width."""
    print("\n=== Experiment 2: Very Wide Portals ===\n")

    results = {}
    num_cells = 128  # Smaller universe for faster computation
    num_generations = 100
    t_exit = 20
    t_enter = 60
    time_gap = t_enter - t_exit

    # Portal widths from small to nearly full width
    widths = [16, 32, 48, 64, 80, 96, 112, 120]

    for width in widths:
        if width >= num_cells - 10:  # Leave some boundary
            continue

        cycles = []
        pre_cycles = []

        for seed in range(500):
            config = SimConfig(
                rule=110,
                portal_width=width,
                init_density=0.5,
                t_enter=t_enter,
                t_exit=t_exit,
                num_cells=num_cells,
                num_generations=num_generations,
                portal_offset=0,  # Center the portal
            )
            sim = TimeTravelSimulator(config)
            result = sim.run(max_trips=10000, seed=seed)

            if result.found_loop:
                cycles.append(result.cycle_length)
                pre_cycles.append(result.pre_cycle_length)

        results[width] = {
            'width_ratio': width / num_cells,
            'mean_cycle': np.mean(cycles) if cycles else 0,
            'max_cycle': max(cycles) if cycles else 0,
            'mean_pre_cycle': np.mean(pre_cycles) if pre_cycles else 0,
            'oscillating': sum(1 for c in cycles if c > 1),
        }

        print(f"Width {width:3d} ({width/num_cells:.0%}): mean_cycle={results[width]['mean_cycle']:.1f}, "
              f"oscillating={results[width]['oscillating']}")

    return results


def experiment_short_time_gap():
    """Very short time gaps to see if oscillations appear."""
    print("\n=== Experiment 3: Very Short Time Gaps ===\n")

    results = {}
    num_cells = 128
    num_generations = 60
    portal_width = 32

    for time_gap in [2, 3, 4, 5, 6, 8, 10]:
        t_exit = 10
        t_enter = t_exit + time_gap

        cycles = []
        pre_cycles = []

        for seed in range(500):
            config = SimConfig(
                rule=110,
                portal_width=portal_width,
                init_density=0.5,
                t_enter=t_enter,
                t_exit=t_exit,
                num_cells=num_cells,
                num_generations=num_generations,
            )
            sim = TimeTravelSimulator(config)
            result = sim.run(max_trips=10000, seed=seed)

            if result.found_loop:
                cycles.append(result.cycle_length)
                pre_cycles.append(result.pre_cycle_length)

        results[time_gap] = {
            'mean_cycle': np.mean(cycles) if cycles else 0,
            'max_cycle': max(cycles) if cycles else 0,
            'oscillating': sum(1 for c in cycles if c > 1),
            'cycle_dist': dict(Counter(cycles)),
        }

        print(f"Time gap {time_gap:2d}: mean_cycle={results[time_gap]['mean_cycle']:.1f}, "
              f"oscillating={results[time_gap]['oscillating']}, "
              f"cycle_dist={results[time_gap]['cycle_dist']}")

    return results


def experiment_full_width_portal():
    """Portal spans entire width - no boundary effects."""
    print("\n=== Experiment 4: Full Width Portal (No Boundaries) ===\n")

    results = {}
    num_generations = 100

    # Universe width equals portal width
    for width in [16, 24, 32, 48, 64]:
        cycles = []
        pre_cycles = []

        for time_gap in [10, 20, 30]:
            t_exit = 10
            t_enter = t_exit + time_gap

            for seed in range(200):
                config = SimConfig(
                    rule=110,
                    portal_width=width,
                    init_density=0.5,
                    t_enter=t_enter,
                    t_exit=t_exit,
                    num_cells=width,  # Full width
                    num_generations=num_generations,
                    portal_offset=0,
                )
                sim = TimeTravelSimulator(config)
                result = sim.run(max_trips=10000, seed=seed)

                if result.found_loop:
                    cycles.append(result.cycle_length)
                    pre_cycles.append(result.pre_cycle_length)

        results[width] = {
            'mean_cycle': np.mean(cycles) if cycles else 0,
            'max_cycle': max(cycles) if cycles else 0,
            'mean_pre_cycle': np.mean(pre_cycles) if pre_cycles else 0,
            'oscillating': sum(1 for c in cycles if c > 1),
            'cycle_dist': dict(Counter(cycles)),
        }

        print(f"Width {width:2d} (full): mean_cycle={results[width]['mean_cycle']:.1f}, "
              f"max_cycle={results[width]['max_cycle']}, "
              f"oscillating={results[width]['oscillating']}, "
              f"cycle_dist={results[width]['cycle_dist']}")

    return results


def experiment_different_rules():
    """Test if different rules produce different behavior."""
    print("\n=== Experiment 5: Different Rules with Various Settings ===\n")

    rules_to_test = [30, 45, 60, 90, 105, 110, 150, 182]

    for rule in rules_to_test:
        cycles = []
        pre_cycles = []

        # Short time gap, moderate portal
        for seed in range(300):
            config = SimConfig(
                rule=rule,
                portal_width=32,
                init_density=0.5,
                t_enter=30,
                t_exit=10,
                num_cells=128,
                num_generations=60,
            )
            sim = TimeTravelSimulator(config)
            result = sim.run(max_trips=10000, seed=seed)

            if result.found_loop:
                cycles.append(result.cycle_length)
                pre_cycles.append(result.pre_cycle_length)

        oscillating = sum(1 for c in cycles if c > 1)
        print(f"Rule {rule:3d}: mean_cycle={np.mean(cycles):.1f}, "
              f"max_cycle={max(cycles) if cycles else 0}, "
              f"oscillating={oscillating}, "
              f"dist={dict(Counter(cycles))}")


def main():
    print("="*60)
    print("BOUNDARY EFFECTS INVESTIGATION")
    print("="*60)
    print()

    experiment_varying_time_gap()
    experiment_varying_portal_width()
    experiment_short_time_gap()
    experiment_full_width_portal()
    experiment_different_rules()


if __name__ == "__main__":
    main()
