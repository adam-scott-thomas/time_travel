#!/usr/bin/env python3
"""
Comprehensive experiment runner for time travel paradox analysis.

This module runs large-scale experiments to answer:
1. How frequently do perfect loops exist?
2. What is the distribution of oscillating loop phases?
3. What is the distribution of iterations before a loop starts?
4. How do these vary with portal width?
5. How do these depend on initialization?
"""

import json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
from tqdm import tqdm

from simulation import SimConfig, SimResult, TimeTravelSimulator


@dataclass
class ExperimentStats:
    """Aggregated statistics from a set of experiments."""
    total_runs: int = 0
    loops_found: int = 0
    perfect_loops: int = 0
    oscillating_loops: int = 0

    # For distributions (stored as lists for JSON serialization)
    cycle_lengths: List[int] = None
    pre_cycle_lengths: List[int] = None
    total_trips: List[int] = None

    def __post_init__(self):
        if self.cycle_lengths is None:
            self.cycle_lengths = []
        if self.pre_cycle_lengths is None:
            self.pre_cycle_lengths = []
        if self.total_trips is None:
            self.total_trips = []

    @property
    def loop_rate(self) -> float:
        return self.loops_found / self.total_runs if self.total_runs > 0 else 0

    @property
    def perfect_loop_rate(self) -> float:
        return self.perfect_loops / self.total_runs if self.total_runs > 0 else 0

    @property
    def oscillating_loop_rate(self) -> float:
        return self.oscillating_loops / self.total_runs if self.total_runs > 0 else 0

    @property
    def mean_cycle_length(self) -> float:
        return np.mean(self.cycle_lengths) if self.cycle_lengths else 0

    @property
    def mean_pre_cycle_length(self) -> float:
        return np.mean(self.pre_cycle_lengths) if self.pre_cycle_lengths else 0

    def add_result(self, result: SimResult):
        self.total_runs += 1
        self.total_trips.append(result.total_trips)

        if result.found_loop:
            self.loops_found += 1
            self.cycle_lengths.append(result.cycle_length)
            self.pre_cycle_lengths.append(result.pre_cycle_length)

            if result.is_perfect_loop:
                self.perfect_loops += 1
            else:
                self.oscillating_loops += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_runs': self.total_runs,
            'loops_found': self.loops_found,
            'perfect_loops': self.perfect_loops,
            'oscillating_loops': self.oscillating_loops,
            'loop_rate': self.loop_rate,
            'perfect_loop_rate': self.perfect_loop_rate,
            'oscillating_loop_rate': self.oscillating_loop_rate,
            'mean_cycle_length': self.mean_cycle_length,
            'mean_pre_cycle_length': self.mean_pre_cycle_length,
            'cycle_lengths': self.cycle_lengths,
            'pre_cycle_lengths': self.pre_cycle_lengths,
            'total_trips': self.total_trips,
        }


def _run_single(args: Tuple[SimConfig, int, int]) -> Tuple[dict, dict]:
    """Worker function for parallel execution."""
    config, max_trips, seed = args
    sim = TimeTravelSimulator(config)
    result = sim.run(max_trips=max_trips, seed=seed)

    # Convert to dicts for pickling
    config_dict = asdict(config)
    result_dict = {
        'cycle_start': result.cycle_start,
        'cycle_length': result.cycle_length,
        'total_trips': result.total_trips,
        'found_loop': result.found_loop,
        'pre_cycle_length': result.pre_cycle_length,
        'is_perfect_loop': result.is_perfect_loop,
    }
    return config_dict, result_dict


class ExperimentRunner:
    """Runs and manages large-scale experiments."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_parameter_sweep(
        self,
        rules: List[int],
        portal_widths: List[int],
        init_densities: List[float],
        reps_per_config: int = 1000,
        max_trips: int = 10000,
        n_workers: Optional[int] = None,
        save_name: str = "sweep_results"
    ) -> Dict[str, Any]:
        """Run a full parameter sweep across rules, widths, and densities."""

        if n_workers is None:
            n_workers = cpu_count()

        # Generate all experiment configurations
        configs = []
        for rule in rules:
            for width in portal_widths:
                for density in init_densities:
                    for rep in range(reps_per_config):
                        config = SimConfig(
                            rule=rule,
                            portal_width=width,
                            init_density=density,
                        )
                        # Use deterministic seed for reproducibility
                        seed = hash((rule, width, density, rep)) % (2**31)
                        configs.append((config, max_trips, seed))

        total_experiments = len(configs)
        print(f"Running {total_experiments:,} experiments with {n_workers} workers...")

        # Run experiments in parallel
        results = []
        start_time = time.time()

        with Pool(n_workers) as pool:
            for result in tqdm(pool.imap(_run_single, configs, chunksize=100),
                               total=total_experiments):
                results.append(result)

        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.1f}s ({total_experiments/elapsed:.1f} experiments/sec)")

        # Aggregate results by different groupings
        aggregated = self._aggregate_results(results)

        # Save results
        self._save_results(results, aggregated, save_name)

        return aggregated

    def _aggregate_results(self, results: List[Tuple[dict, dict]]) -> Dict[str, Any]:
        """Aggregate results by various groupings."""

        # Group by different parameters
        by_rule = {}
        by_width = {}
        by_density = {}
        by_rule_width = {}
        overall = ExperimentStats()

        for config_dict, result_dict in results:
            rule = config_dict['rule']
            width = config_dict['portal_width']
            density = config_dict['init_density']

            # Create SimResult-like object
            result = type('Result', (), result_dict)()

            # Overall
            overall.add_result(result)

            # By rule
            if rule not in by_rule:
                by_rule[rule] = ExperimentStats()
            by_rule[rule].add_result(result)

            # By width
            if width not in by_width:
                by_width[width] = ExperimentStats()
            by_width[width].add_result(result)

            # By density
            if density not in by_density:
                by_density[density] = ExperimentStats()
            by_density[density].add_result(result)

            # By rule and width combination
            key = (rule, width)
            if key not in by_rule_width:
                by_rule_width[key] = ExperimentStats()
            by_rule_width[key].add_result(result)

        return {
            'overall': overall.to_dict(),
            'by_rule': {k: v.to_dict() for k, v in by_rule.items()},
            'by_width': {k: v.to_dict() for k, v in by_width.items()},
            'by_density': {k: v.to_dict() for k, v in by_density.items()},
            'by_rule_width': {f"{k[0]}_{k[1]}": v.to_dict() for k, v in by_rule_width.items()},
        }

    def _save_results(self, raw_results: List, aggregated: Dict, save_name: str):
        """Save results to disk."""
        # Save raw results
        raw_path = self.output_dir / f"{save_name}_raw.pkl"
        with open(raw_path, 'wb') as f:
            pickle.dump(raw_results, f)
        print(f"Saved raw results to {raw_path}")

        # Save aggregated results as JSON for easy reading
        json_path = self.output_dir / f"{save_name}_aggregated.json"
        with open(json_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
        print(f"Saved aggregated results to {json_path}")


def run_quick_test():
    """Quick test with small parameters."""
    runner = ExperimentRunner()
    results = runner.run_parameter_sweep(
        rules=[30, 110],
        portal_widths=[8, 10, 12],
        init_densities=[0.5],
        reps_per_config=100,
        max_trips=1000,
    )

    print("\n=== Quick Test Results ===")
    print(f"Overall loop rate: {results['overall']['loop_rate']:.2%}")
    print(f"Perfect loop rate: {results['overall']['perfect_loop_rate']:.2%}")
    print(f"Mean cycle length: {results['overall']['mean_cycle_length']:.2f}")


def run_full_experiment():
    """Run the full comprehensive experiment."""
    runner = ExperimentRunner()

    # Interesting chaotic rules identified from prior research
    rules = [30, 45, 73, 97, 110, 137, 161, 165, 169]

    # Portal widths to test - from narrow to wide
    # Each additional bit doubles the state space
    portal_widths = list(range(6, 22, 2))  # 6, 8, 10, ..., 20

    # Initial densities
    init_densities = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("=== Full Experiment Configuration ===")
    print(f"Rules: {rules}")
    print(f"Portal widths: {portal_widths}")
    print(f"Init densities: {init_densities}")
    print(f"Reps per config: 500")
    total = len(rules) * len(portal_widths) * len(init_densities) * 500
    print(f"Total experiments: {total:,}")
    print()

    results = runner.run_parameter_sweep(
        rules=rules,
        portal_widths=portal_widths,
        init_densities=init_densities,
        reps_per_config=500,
        max_trips=50000,  # Higher limit for wider portals
        save_name="full_experiment"
    )

    print("\n=== Summary ===")
    print(f"Overall loop rate: {results['overall']['loop_rate']:.2%}")
    print(f"Perfect loop rate: {results['overall']['perfect_loop_rate']:.2%}")
    print(f"Oscillating loop rate: {results['overall']['oscillating_loop_rate']:.2%}")
    print(f"Mean cycle length: {results['overall']['mean_cycle_length']:.2f}")
    print(f"Mean pre-cycle length: {results['overall']['mean_pre_cycle_length']:.2f}")


def run_width_scaling_experiment():
    """Focused experiment on how loop properties scale with portal width."""
    runner = ExperimentRunner()

    # Use a single well-behaved chaotic rule
    rules = [110]

    # Fine-grained width sweep
    portal_widths = list(range(4, 25))  # 4 to 24

    # Single density for controlled comparison
    init_densities = [0.5]

    print("=== Width Scaling Experiment ===")
    total = len(rules) * len(portal_widths) * len(init_densities) * 2000
    print(f"Total experiments: {total:,}")

    results = runner.run_parameter_sweep(
        rules=rules,
        portal_widths=portal_widths,
        init_densities=init_densities,
        reps_per_config=2000,
        max_trips=100000,
        save_name="width_scaling"
    )

    print("\n=== Width Scaling Results ===")
    for width in sorted(results['by_width'].keys()):
        stats = results['by_width'][width]
        print(f"Width {width:2d}: loop_rate={stats['loop_rate']:.2%}, "
              f"mean_cycle={stats['mean_cycle_length']:.1f}, "
              f"mean_pre_cycle={stats['mean_pre_cycle_length']:.1f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            run_quick_test()
        elif sys.argv[1] == "width":
            run_width_scaling_experiment()
        elif sys.argv[1] == "full":
            run_full_experiment()
    else:
        print("Usage: python experiments.py [quick|width|full]")
        print("  quick - Quick test with small parameters")
        print("  width - Width scaling experiment")
        print("  full  - Full comprehensive experiment")
