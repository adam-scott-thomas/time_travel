#!/usr/bin/env python3
"""
Time Travel Paradox Simulation using 1D Cellular Automata

This module provides efficient simulation of 1D cellular automata with "time travel"
where cells can be sent back in time, creating potential paradox loops.

Key concepts:
- Universe: 2D array where rows are time steps, columns are cell positions
- Time portal: A region that captures cells at t_enter and sends them back to t_exit
- Perfect loop: Same pattern sent back results in identical pattern (cycle_length=1)
- Oscillating loop: Pattern A -> B -> C -> ... -> A (cycle_length > 1)
- Pre-cycle length: Number of iterations before entering a loop (cycle_start - 1)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
from collections import namedtuple


@dataclass
class SimConfig:
    """Configuration for a single simulation run."""
    rule: int = 110           # Wolfram rule number (0-255)
    init_density: float = 0.5  # Initial cell density (0.0-1.0) for random init
    t_enter: int = 80         # Time step when cells enter portal
    t_exit: int = 40          # Time step when cells exit portal (in the past)
    portal_width: int = 32    # Width of portal in cells
    num_cells: int = 256      # Total number of cells in the universe
    num_generations: int = 144  # Total time steps before wrapping
    center_init: bool = False  # If True, start with single cell in center
    portal_offset: int = 5    # Offset from center for portal position


@dataclass
class SimResult:
    """Results from a single simulation run."""
    cycle_start: int          # Trip number when loop pattern first appeared (1-indexed)
    cycle_length: int         # Number of trips in the cycle (1 = perfect loop)
    total_trips: int          # Total trips before loop detected
    found_loop: bool          # Whether a loop was found

    @property
    def pre_cycle_length(self) -> int:
        """Number of trips before the loop started."""
        return self.cycle_start - 1

    @property
    def is_perfect_loop(self) -> bool:
        """True if this is a perfect/stable time loop (cycle_length=1)."""
        return self.found_loop and self.cycle_length == 1


def rule_to_lookup(rule_number: int) -> np.ndarray:
    """Convert Wolfram rule number to lookup table.

    For a 3-cell neighborhood, there are 8 possible patterns (000 to 111).
    The rule number's binary representation determines the output for each pattern.
    """
    # Convert rule to 8-bit binary representation
    lookup = np.zeros(8, dtype=np.int8)
    for i in range(8):
        lookup[7 - i] = (rule_number >> i) & 1
    return lookup


class TimeTravelSimulator:
    """Efficient 1D Cellular Automata simulator with time travel support."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.rule_lookup = rule_to_lookup(config.rule)
        self.reset()

    def reset(self, seed: Optional[int] = None):
        """Reset the simulation to initial state."""
        if seed is not None:
            np.random.seed(seed)

        cfg = self.config

        # Initialize universe
        self.universe = np.zeros((cfg.num_generations, cfg.num_cells), dtype=np.int8)

        # Set initial conditions
        if cfg.center_init:
            self.universe[0, cfg.num_cells // 2] = 1
        else:
            self.universe[0] = (np.random.rand(cfg.num_cells) < cfg.init_density).astype(np.int8)

        # Portal positions (centered with offset)
        self.portal_x = cfg.num_cells // 2 + cfg.portal_offset

        # State tracking
        self.history: Dict[tuple, int] = {}  # pattern -> trip number when first seen
        self.trips = 0
        self.current_gen = 0
        self.result: Optional[SimResult] = None

    def _evolve_row(self, t: int) -> np.ndarray:
        """Compute the next generation using vectorized operations."""
        row = self.universe[t]
        cfg = self.config

        # Get left, center, right neighborhoods (with wrapping)
        left = np.roll(row, 1)
        center = row
        right = np.roll(row, -1)

        # Compute index into rule lookup table
        # Pattern 111 = 7, 110 = 6, ..., 000 = 0
        indices = 4 * left + 2 * center + right

        # Apply rule
        new_row = self.rule_lookup[7 - indices]
        return new_row

    def _check_portal(self) -> Optional[SimResult]:
        """Check if we've entered the portal and handle time travel."""
        cfg = self.config

        if self.current_gen != cfg.t_enter:
            return None

        # Extract portal contents from next generation
        portal_slice = slice(self.portal_x, self.portal_x + cfg.portal_width)
        portal_contents = tuple(self.universe[self.current_gen + 1, portal_slice])

        self.trips += 1

        # Check for loop
        if portal_contents in self.history:
            first_trip = self.history[portal_contents]
            return SimResult(
                cycle_start=first_trip,
                cycle_length=self.trips - first_trip,
                total_trips=self.trips,
                found_loop=True
            )

        # Record this pattern
        self.history[portal_contents] = self.trips

        # Send pattern back in time
        self.universe[cfg.t_exit, portal_slice] = portal_contents

        # Reset to continue from exit point
        self.current_gen = cfg.t_exit

        return None

    def step(self) -> bool:
        """Advance simulation by one step. Returns True if loop found."""
        if self.result is not None:
            return True

        cfg = self.config

        # Evolve current generation
        if self.current_gen < cfg.num_generations - 1:
            self.universe[self.current_gen + 1] = self._evolve_row(self.current_gen)

        # Check for portal entry and time travel
        result = self._check_portal()
        if result is not None:
            self.result = result
            return True

        self.current_gen += 1
        return False

    def run(self, max_trips: int = 10000, seed: Optional[int] = None) -> SimResult:
        """Run simulation until loop found or max_trips exceeded."""
        self.reset(seed)

        while self.trips < max_trips:
            if self.step():
                return self.result

        # No loop found within max_trips
        return SimResult(
            cycle_start=0,
            cycle_length=0,
            total_trips=self.trips,
            found_loop=False
        )

    def get_universe_snapshot(self) -> np.ndarray:
        """Get a copy of the current universe state."""
        return self.universe.copy()


def run_single_experiment(config: SimConfig, max_trips: int = 10000,
                          seed: Optional[int] = None) -> Tuple[SimConfig, SimResult]:
    """Run a single experiment and return config + result."""
    sim = TimeTravelSimulator(config)
    result = sim.run(max_trips=max_trips, seed=seed)
    return config, result


def run_batch_experiments(configs: List[SimConfig], max_trips: int = 10000,
                          n_workers: Optional[int] = None,
                          show_progress: bool = True) -> List[Tuple[SimConfig, SimResult]]:
    """Run multiple experiments in parallel."""
    from multiprocessing import Pool, cpu_count
    from functools import partial

    if n_workers is None:
        n_workers = cpu_count()

    runner = partial(run_single_experiment, max_trips=max_trips)

    results = []
    with Pool(n_workers) as pool:
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(pool.imap(runner, configs), total=len(configs))
        else:
            iterator = pool.imap(runner, configs)

        for result in iterator:
            results.append(result)

    return results


if __name__ == "__main__":
    # Quick test
    config = SimConfig(rule=110, init_density=0.5, portal_width=32)
    sim = TimeTravelSimulator(config)
    result = sim.run(max_trips=1000)
    print(f"Config: Rule {config.rule}, Width {config.portal_width}")
    print(f"Result: {result}")
    if result.found_loop:
        print(f"  Pre-cycle length: {result.pre_cycle_length}")
        print(f"  Perfect loop: {result.is_perfect_loop}")
