#! /usr/bin/python3

# Copyright 2026 Adam Scott Thomas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyze experimental results from time travel cellular automata simulations.

This script processes pickled experiment results to categorize Wolfram rules
by their temporal loop behavior patterns.
"""

import pickle
from typing import Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict

from time_cell import Result, Config


def load_experimental_data(filename: str) -> List[Tuple[Config, Optional[Result]]]:
    """Load experimental results from pickle file.

    Args:
        filename: Path to pickle file containing experiment results

    Returns:
        List of (config, result) tuples from experiments
    """
    data = []
    try:
        with open(filename, "rb") as f:
            while True:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
    except FileNotFoundError:
        print(f"Data file {filename} not found. Run experiments first with run_experiments.py")
        return []

    return data


def analyze_rule_behavior(data: List[Tuple[Config, Optional[Result]]]) -> Dict[int, Dict[str, int]]:
    """Analyze temporal loop behavior by Wolfram rule.

    Args:
        data: List of experimental results

    Returns:
        Dictionary mapping rule numbers to statistics
    """
    stats: DefaultDict[int, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))

    for config, result in data:
        stats[config.rule]["count"] += 1
        if result is None:
            stats[config.rule]["no_loops"] += 1
        else:
            stats[config.rule]["total_cycle_length"] += result.cycle_length
            stats[config.rule]["loops_found"] += 1

    return dict(stats)


def categorize_rules(stats: Dict[int, Dict[str, int]]) -> Dict[str, List[int]]:
    """Categorize rules by their temporal behavior patterns.

    Args:
        stats: Rule statistics from analyze_rule_behavior

    Returns:
        Dictionary mapping behavior categories to rule lists
    """
    categories = {
        "no_time_travel": [],     # Rules that never form loops
        "medium_time_travel": [], # Rules with moderate loop formation
        "high_time_travel": []    # Rules with frequent/complex loops
    }

    for rule, data in stats.items():
        if data["count"] == 0:
            continue

        loop_rate = data["loops_found"] / data["count"]
        avg_cycle_length = data["total_cycle_length"] / max(data["loops_found"], 1)

        if loop_rate < 0.1:
            categories["no_time_travel"].append(rule)
        elif loop_rate < 0.5 or avg_cycle_length < 5:
            categories["medium_time_travel"].append(rule)
        else:
            categories["high_time_travel"].append(rule)

    return categories


def print_analysis_report(stats: Dict[int, Dict[str, int]], categories: Dict[str, List[int]]) -> None:
    """Print comprehensive analysis report.

    Args:
        stats: Rule statistics
        categories: Categorized rules by behavior
    """
    print("=" * 80)
    print("TIME TRAVEL CELLULAR AUTOMATA - ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nTotal rules analyzed: {len(stats)}")
    print(f"Total experiments: {sum(data['count'] for data in stats.values())}")

    print("\nDetailed Rule Statistics:")
    print("-" * 60)
    print("Rule | Experiments | Loops Found | No Loops | Avg Cycle Length")
    print("-" * 60)

    for rule in sorted(stats.keys()):
        data = stats[rule]
        avg_cycle = data["total_cycle_length"] / max(data["loops_found"], 1)
        loop_rate = data["loops_found"] / data["count"] * 100

        print(f"{rule:4d} | {data['count']:11d} | {data['loops_found']:11d} | "
              f"{data['no_loops']:8d} | {avg_cycle:14.1f}")

    print("\nRule Categories by Temporal Behavior:")
    print("-" * 40)

    for category, rules in categories.items():
        print(f"\n{category.replace('_', ' ').title()}: {len(rules)} rules")
        if rules:
            print(f"  Rules: {sorted(rules)}")

    print("\nRecommended Rules for Further Study:")
    interesting_rules = [30, 45, 73, 97, 110, 137, 161, 165, 169]
    print(f"  Diverse behavior set: {interesting_rules}")


def main() -> None:
    """Main analysis workflow."""
    # Load experimental data
    data = load_experimental_data("main_rules.p")
    if not data:
        return

    print(f"Loaded {len(data)} experimental results")

    # Analyze rule behavior
    stats = analyze_rule_behavior(data)
    categories = categorize_rules(stats)

    # Generate report
    print_analysis_report(stats, categories)


if __name__ == "__main__":
    main()
