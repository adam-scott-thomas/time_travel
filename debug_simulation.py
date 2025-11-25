#!/usr/bin/env python3
"""
Debug the simulation to understand why we always get perfect loops.
"""

import numpy as np
from simulation import SimConfig, TimeTravelSimulator


def trace_simulation():
    """Trace through a simulation step by step."""
    config = SimConfig(
        rule=110,
        portal_width=8,
        init_density=0.5,
        t_enter=20,
        t_exit=10,
        num_cells=32,
        num_generations=40,
        portal_offset=0,  # Center portal
    )

    sim = TimeTravelSimulator(config)
    sim.reset(seed=42)

    print("=== Initial State ===")
    print(f"Portal x position: {sim.portal_x}")
    print(f"Portal slice: {sim.portal_x} to {sim.portal_x + config.portal_width}")
    print(f"Num cells: {config.num_cells}")
    print()

    # Show initial row
    print(f"Row 0: {''.join(str(x) for x in sim.universe[0])}")

    # Evolve to t_exit
    while sim.current_gen < config.t_exit:
        sim.step()

    print(f"\n=== At t_exit ({config.t_exit}) ===")
    print(f"Row {config.t_exit}: {''.join(str(x) for x in sim.universe[config.t_exit])}")
    portal_at_exit = list(sim.universe[config.t_exit, sim.portal_x:sim.portal_x + config.portal_width])
    print(f"Portal region: {''.join(str(x) for x in portal_at_exit)}")

    # Evolve to t_enter
    while sim.current_gen < config.t_enter:
        sim.step()

    print(f"\n=== At t_enter ({config.t_enter}) before time travel ===")
    print(f"Row {config.t_enter}: {''.join(str(x) for x in sim.universe[config.t_enter])}")
    print(f"Row {config.t_enter+1}: {''.join(str(x) for x in sim.universe[config.t_enter+1])}")
    portal_contents_1 = list(sim.universe[config.t_enter+1, sim.portal_x:sim.portal_x + config.portal_width])
    print(f"Portal contents (trip 1): {''.join(str(x) for x in portal_contents_1)}")

    # Trigger time travel
    print("\n=== Time Travel Triggered ===")
    sim.step()  # This should trigger time travel and record pattern

    print(f"Current gen after step: {sim.current_gen}")
    print(f"Trips: {sim.trips}")
    print(f"History keys: {len(sim.history)}")

    # Show row at t_exit after time travel
    print(f"\n=== Row at t_exit after time travel ===")
    print(f"Row {config.t_exit}: {''.join(str(x) for x in sim.universe[config.t_exit])}")
    new_portal_at_exit = list(sim.universe[config.t_exit, sim.portal_x:sim.portal_x + config.portal_width])
    print(f"Portal region: {''.join(str(x) for x in new_portal_at_exit)}")

    # Check if outside region changed
    outside_left = list(sim.universe[config.t_exit, :sim.portal_x])
    outside_right = list(sim.universe[config.t_exit, sim.portal_x + config.portal_width:])
    print(f"Outside left: {''.join(str(x) for x in outside_left)}")
    print(f"Outside right: {''.join(str(x) for x in outside_right)}")

    # Continue to t_enter again
    while sim.current_gen < config.t_enter:
        sim.step()

    print(f"\n=== At t_enter ({config.t_enter}) on trip 2 ===")
    print(f"Row {config.t_enter}: {''.join(str(x) for x in sim.universe[config.t_enter])}")
    print(f"Row {config.t_enter+1}: {''.join(str(x) for x in sim.universe[config.t_enter+1])}")
    portal_contents_2 = list(sim.universe[config.t_enter+1, sim.portal_x:sim.portal_x + config.portal_width])
    print(f"Portal contents (trip 2): {''.join(str(x) for x in portal_contents_2)}")

    # Compare
    print(f"\n=== Comparison ===")
    print(f"Trip 1: {''.join(str(x) for x in portal_contents_1)}")
    print(f"Trip 2: {''.join(str(x) for x in portal_contents_2)}")
    print(f"Match: {portal_contents_1 == portal_contents_2}")


def test_full_width_portal():
    """Test with portal covering full width."""
    print("\n" + "="*60)
    print("FULL WIDTH PORTAL TEST")
    print("="*60 + "\n")

    # Make portal cover entire width
    width = 16
    config = SimConfig(
        rule=110,
        portal_width=width,
        init_density=0.5,
        t_enter=20,
        t_exit=10,
        num_cells=width,  # Same as portal width
        num_generations=40,
        portal_offset=-width//2,  # Center at 0
    )

    sim = TimeTravelSimulator(config)
    sim.reset(seed=42)

    print(f"Portal x position: {sim.portal_x}")
    print(f"Portal slice: {sim.portal_x} to {sim.portal_x + config.portal_width}")
    print(f"Num cells: {config.num_cells}")

    # Check if portal covers full width
    portal_slice = slice(sim.portal_x, sim.portal_x + config.portal_width)
    full_slice = slice(0, config.num_cells)
    print(f"Portal covers indices: {list(range(config.num_cells)[portal_slice])}")
    print(f"Full width indices: {list(range(config.num_cells)[full_slice])}")

    # Run a full simulation
    result = sim.run(max_trips=1000, seed=42)
    print(f"\nResult: {result}")


def test_modified_time_travel():
    """Test time travel where we ALSO clear the outside region."""
    print("\n" + "="*60)
    print("MODIFIED TIME TRAVEL (clear outside region)")
    print("="*60 + "\n")

    from dataclasses import dataclass
    from typing import Optional, Dict

    @dataclass
    class ModifiedSimResult:
        cycle_start: int
        cycle_length: int
        total_trips: int
        found_loop: bool

        @property
        def pre_cycle_length(self) -> int:
            return self.cycle_start - 1

        @property
        def is_perfect_loop(self) -> bool:
            return self.found_loop and self.cycle_length == 1

    class ModifiedTimeTravelSimulator:
        """Time travel where we reset ENTIRE row at t_exit."""

        def __init__(self, config: SimConfig):
            self.config = config
            from simulation import rule_to_lookup
            self.rule_lookup = rule_to_lookup(config.rule)
            self.reset()

        def reset(self, seed: Optional[int] = None):
            if seed is not None:
                np.random.seed(seed)

            cfg = self.config
            self.universe = np.zeros((cfg.num_generations, cfg.num_cells), dtype=np.int8)

            if cfg.center_init:
                self.universe[0, cfg.num_cells // 2] = 1
            else:
                self.universe[0] = (np.random.rand(cfg.num_cells) < cfg.init_density).astype(np.int8)

            self.portal_x = max(0, cfg.num_cells // 2 + cfg.portal_offset)
            self.history: Dict[tuple, int] = {}
            self.trips = 0
            self.current_gen = 0
            self.result = None

        def _evolve_row(self, t: int) -> np.ndarray:
            row = self.universe[t]
            left = np.roll(row, 1)
            center = row
            right = np.roll(row, -1)
            indices = 4 * left + 2 * center + right
            new_row = self.rule_lookup[7 - indices]
            return new_row

        def _check_portal(self):
            cfg = self.config
            if self.current_gen != cfg.t_enter:
                return None

            portal_slice = slice(self.portal_x, self.portal_x + cfg.portal_width)
            portal_contents = tuple(self.universe[self.current_gen + 1, portal_slice])

            self.trips += 1

            if portal_contents in self.history:
                first_trip = self.history[portal_contents]
                return ModifiedSimResult(
                    cycle_start=first_trip,
                    cycle_length=self.trips - first_trip,
                    total_trips=self.trips,
                    found_loop=True
                )

            self.history[portal_contents] = self.trips

            # KEY MODIFICATION: Reset entire row at t_exit to zeros, then insert portal
            self.universe[cfg.t_exit] = 0  # Clear entire row
            self.universe[cfg.t_exit, portal_slice] = portal_contents

            self.current_gen = cfg.t_exit
            return None

        def step(self) -> bool:
            if self.result is not None:
                return True

            cfg = self.config
            if self.current_gen < cfg.num_generations - 1:
                self.universe[self.current_gen + 1] = self._evolve_row(self.current_gen)

            result = self._check_portal()
            if result is not None:
                self.result = result
                return True

            self.current_gen += 1
            return False

        def run(self, max_trips: int = 10000, seed: Optional[int] = None):
            self.reset(seed)
            while self.trips < max_trips:
                if self.step():
                    return self.result
            return ModifiedSimResult(
                cycle_start=0, cycle_length=0, total_trips=self.trips, found_loop=False
            )

    # Test the modified simulator
    from collections import Counter

    cycles = []
    pre_cycles = []

    for seed in range(500):
        config = SimConfig(
            rule=110,
            portal_width=16,
            init_density=0.5,
            t_enter=30,
            t_exit=10,
            num_cells=64,
            num_generations=60,
        )
        sim = ModifiedTimeTravelSimulator(config)
        result = sim.run(max_trips=10000, seed=seed)

        if result.found_loop:
            cycles.append(result.cycle_length)
            pre_cycles.append(result.pre_cycle_length)

    print(f"Results with modified time travel (cleared outside):")
    print(f"  Runs: {len(cycles)}")
    print(f"  Mean cycle: {np.mean(cycles):.2f}")
    print(f"  Max cycle: {max(cycles) if cycles else 0}")
    print(f"  Oscillating: {sum(1 for c in cycles if c > 1)}")
    print(f"  Cycle dist: {dict(Counter(cycles))}")


def test_randomize_outside():
    """Time travel where we randomize the outside region each trip."""
    print("\n" + "="*60)
    print("RANDOMIZED OUTSIDE REGION")
    print("="*60 + "\n")

    from dataclasses import dataclass
    from typing import Optional, Dict

    @dataclass
    class ModifiedSimResult:
        cycle_start: int
        cycle_length: int
        total_trips: int
        found_loop: bool

        @property
        def pre_cycle_length(self) -> int:
            return self.cycle_start - 1

        @property
        def is_perfect_loop(self) -> bool:
            return self.found_loop and self.cycle_length == 1

    class RandomizedTimeTravelSimulator:
        """Time travel where we randomize cells outside the portal."""

        def __init__(self, config: SimConfig):
            self.config = config
            from simulation import rule_to_lookup
            self.rule_lookup = rule_to_lookup(config.rule)
            self.reset()

        def reset(self, seed: Optional[int] = None):
            if seed is not None:
                np.random.seed(seed)
                self.rng = np.random.RandomState(seed)
            else:
                self.rng = np.random.RandomState()

            cfg = self.config
            self.universe = np.zeros((cfg.num_generations, cfg.num_cells), dtype=np.int8)

            if cfg.center_init:
                self.universe[0, cfg.num_cells // 2] = 1
            else:
                self.universe[0] = (self.rng.rand(cfg.num_cells) < cfg.init_density).astype(np.int8)

            self.portal_x = max(0, cfg.num_cells // 2 + cfg.portal_offset)
            self.history: Dict[tuple, int] = {}
            self.trips = 0
            self.current_gen = 0
            self.result = None

        def _evolve_row(self, t: int) -> np.ndarray:
            row = self.universe[t]
            left = np.roll(row, 1)
            center = row
            right = np.roll(row, -1)
            indices = 4 * left + 2 * center + right
            new_row = self.rule_lookup[7 - indices]
            return new_row

        def _check_portal(self):
            cfg = self.config
            if self.current_gen != cfg.t_enter:
                return None

            portal_slice = slice(self.portal_x, self.portal_x + cfg.portal_width)
            portal_contents = tuple(self.universe[self.current_gen + 1, portal_slice])

            self.trips += 1

            if portal_contents in self.history:
                first_trip = self.history[portal_contents]
                return ModifiedSimResult(
                    cycle_start=first_trip,
                    cycle_length=self.trips - first_trip,
                    total_trips=self.trips,
                    found_loop=True
                )

            self.history[portal_contents] = self.trips

            # KEY MODIFICATION: Randomize entire row at t_exit, then insert portal
            self.universe[cfg.t_exit] = (self.rng.rand(cfg.num_cells) < cfg.init_density).astype(np.int8)
            self.universe[cfg.t_exit, portal_slice] = portal_contents

            self.current_gen = cfg.t_exit
            return None

        def step(self) -> bool:
            if self.result is not None:
                return True

            cfg = self.config
            if self.current_gen < cfg.num_generations - 1:
                self.universe[self.current_gen + 1] = self._evolve_row(self.current_gen)

            result = self._check_portal()
            if result is not None:
                self.result = result
                return True

            self.current_gen += 1
            return False

        def run(self, max_trips: int = 10000, seed: Optional[int] = None):
            self.reset(seed)
            while self.trips < max_trips:
                if self.step():
                    return self.result
            return ModifiedSimResult(
                cycle_start=0, cycle_length=0, total_trips=self.trips, found_loop=False
            )

    # Test the randomized simulator
    from collections import Counter

    cycles = []
    pre_cycles = []

    for seed in range(500):
        config = SimConfig(
            rule=110,
            portal_width=16,
            init_density=0.5,
            t_enter=30,
            t_exit=10,
            num_cells=64,
            num_generations=60,
        )
        sim = RandomizedTimeTravelSimulator(config)
        result = sim.run(max_trips=10000, seed=seed)

        if result.found_loop:
            cycles.append(result.cycle_length)
            pre_cycles.append(result.pre_cycle_length)

    print(f"Results with randomized outside region:")
    print(f"  Runs: {len(cycles)}")
    print(f"  Mean cycle: {np.mean(cycles):.2f}")
    print(f"  Max cycle: {max(cycles) if cycles else 0}")
    print(f"  Mean pre-cycle: {np.mean(pre_cycles):.2f}")
    print(f"  Max pre-cycle: {max(pre_cycles) if pre_cycles else 0}")
    print(f"  Oscillating: {sum(1 for c in cycles if c > 1)}")
    print(f"  Cycle dist (first 20): {dict(list(Counter(cycles).items())[:20])}")


if __name__ == "__main__":
    trace_simulation()
    test_full_width_portal()
    test_modified_time_travel()
    test_randomize_outside()
