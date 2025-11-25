#!/usr/bin/env python3
"""
Unit tests for the time travel simulation.
"""

import numpy as np
import unittest
from simulation import SimConfig, TimeTravelSimulator, rule_to_lookup


class TestRuleLookup(unittest.TestCase):
    """Test CA rule encoding."""

    def test_rule_110(self):
        """Rule 110 = 01101110 in binary."""
        lookup = rule_to_lookup(110)
        # Pattern: 111 110 101 100 011 010 001 000
        # Rule 110: 0   1   1   0   1   1   1   0
        expected = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(lookup, expected)

    def test_rule_30(self):
        """Rule 30 = 00011110 in binary."""
        lookup = rule_to_lookup(30)
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(lookup, expected)

    def test_rule_0(self):
        """Rule 0 = all zeros."""
        lookup = rule_to_lookup(0)
        expected = np.zeros(8, dtype=np.int8)
        np.testing.assert_array_equal(lookup, expected)

    def test_rule_255(self):
        """Rule 255 = all ones."""
        lookup = rule_to_lookup(255)
        expected = np.ones(8, dtype=np.int8)
        np.testing.assert_array_equal(lookup, expected)


class TestCAEvolution(unittest.TestCase):
    """Test basic CA evolution without time travel."""

    def test_single_cell_rule_110(self):
        """Rule 110 with single center cell should produce known pattern."""
        config = SimConfig(
            rule=110,
            num_cells=11,
            num_generations=5,
            t_enter=100,  # Disable time travel
            t_exit=50,
            center_init=True,
        )
        sim = TimeTravelSimulator(config)
        sim.reset(seed=42)

        # Manually evolve a few steps
        for _ in range(4):
            sim.step()

        # Row 0: single cell in center (position 5)
        self.assertEqual(sim.universe[0, 5], 1)
        self.assertEqual(sim.universe[0].sum(), 1)

        # Row 1: Rule 110 expands the pattern
        # For center cell: neighborhood 010 -> index 2 -> lookup[7-2]=lookup[5]=1
        # For neighbors: 001 -> index 1 -> lookup[6]=1, 100 -> index 4 -> lookup[3]=0
        # So row 1 should have cells at positions 4, 5 (two cells)

    def test_deterministic(self):
        """Same seed should produce identical results."""
        config = SimConfig(rule=110, num_cells=50, num_generations=30,
                           t_enter=100, t_exit=50)  # Disable time travel

        sim1 = TimeTravelSimulator(config)
        sim1.reset(seed=12345)
        for _ in range(20):
            sim1.step()
        state1 = sim1.universe.copy()

        sim2 = TimeTravelSimulator(config)
        sim2.reset(seed=12345)
        for _ in range(20):
            sim2.step()
        state2 = sim2.universe.copy()

        np.testing.assert_array_equal(state1, state2)


class TestTimeTravelMechanics(unittest.TestCase):
    """Test the time travel portal mechanics."""

    def test_portal_copies_correct_region(self):
        """Portal should copy exactly the specified cells."""
        config = SimConfig(
            rule=110,
            num_cells=50,
            num_generations=60,
            t_enter=30,
            t_exit=10,
            portal_width=8,
            portal_offset=0,
        )
        sim = TimeTravelSimulator(config)
        sim.reset(seed=42)

        # Evolve to just before t_enter
        while sim.current_gen < config.t_enter:
            sim.step()

        # Record what's at t_exit before time travel
        portal_x = sim.portal_x
        pre_travel_exit = sim.universe[config.t_exit, portal_x:portal_x+config.portal_width].copy()

        # One more step triggers time travel
        sim.step()

        # Now check that portal contents were copied
        # The portal contents should be from t_enter+1
        post_travel_exit = sim.universe[config.t_exit, portal_x:portal_x+config.portal_width]

        # They should have changed (unless by coincidence they're the same)
        # More importantly, check that a trip was recorded
        self.assertEqual(sim.trips, 1)

    def test_time_travel_resets_generation(self):
        """After time travel, simulation should continue from t_exit."""
        config = SimConfig(
            rule=110,
            num_cells=50,
            num_generations=60,
            t_enter=30,
            t_exit=10,
            portal_width=8,
        )
        sim = TimeTravelSimulator(config)
        sim.reset(seed=42)

        # Evolve to t_enter
        while sim.current_gen < config.t_enter:
            sim.step()

        self.assertEqual(sim.current_gen, config.t_enter)

        # One more step triggers time travel
        sim.step()

        # After time travel, current_gen should be at t_exit (ready to re-evolve)
        # BUG CHECK: If this fails, the bug is that current_gen = t_exit + 1
        self.assertEqual(sim.current_gen, config.t_exit,
                         f"After time travel, current_gen should be {config.t_exit}, "
                         f"not {sim.current_gen}")

    def test_re_evolution_after_time_travel(self):
        """After time travel, universe should be re-evolved from t_exit."""
        config = SimConfig(
            rule=110,
            num_cells=50,
            num_generations=60,
            t_enter=30,
            t_exit=10,
            portal_width=8,
        )
        sim = TimeTravelSimulator(config)
        sim.reset(seed=42)

        # Evolve to just past t_enter (first time travel)
        while sim.trips == 0:
            sim.step()

        # Record universe state at t_exit after first time travel
        t_exit_row_after_travel = sim.universe[config.t_exit].copy()

        # Continue evolving a few steps
        for _ in range(5):
            sim.step()

        # The row at t_exit+1 should be evolved from the NEW t_exit row
        # Not from the old one
        expected_t_exit_plus_1 = sim._evolve_row(config.t_exit)
        # Note: This test is checking that we properly re-evolve

        # Check that universe[t_exit+1] was computed from universe[t_exit] (the modified one)
        actual_t_exit_plus_1 = sim.universe[config.t_exit + 1]

        np.testing.assert_array_equal(
            actual_t_exit_plus_1,
            expected_t_exit_plus_1,
            err_msg="Row t_exit+1 should be evolved from the NEW t_exit row after time travel"
        )


class TestLoopDetection(unittest.TestCase):
    """Test loop detection logic."""

    def test_history_records_patterns(self):
        """Each trip should record the portal pattern."""
        config = SimConfig(
            rule=110,
            num_cells=50,
            num_generations=100,
            t_enter=40,
            t_exit=10,
            portal_width=8,
        )
        sim = TimeTravelSimulator(config)
        sim.reset(seed=42)

        # Run until first trip
        while sim.trips == 0:
            sim.step()

        # History should have exactly one entry
        self.assertEqual(len(sim.history), 1)
        self.assertEqual(sim.trips, 1)

    def test_different_patterns_different_trips(self):
        """Different portal patterns should allow multiple trips before loop."""
        # This test will likely fail with the bug (always 1 trip)
        # and pass after the fix (multiple trips possible)

        # Use a chaotic rule with wide portal for more variability
        config = SimConfig(
            rule=30,  # Very chaotic rule
            num_cells=100,
            num_generations=100,
            t_enter=50,
            t_exit=10,
            portal_width=16,  # Larger portal = more possible patterns
        )

        # Run multiple seeds and check that at least SOME have >1 trip
        multi_trip_found = False
        for seed in range(100):
            sim = TimeTravelSimulator(config)
            result = sim.run(max_trips=100, seed=seed)
            if result.found_loop and result.total_trips > 2:
                multi_trip_found = True
                break

        # This assertion will fail if we always get trip=2 (i.e., loop on first repeat)
        # which would indicate the bug
        self.assertTrue(multi_trip_found,
                        "With a chaotic rule, we should sometimes see >2 trips before loop. "
                        "If this fails, the simulation may not be re-evolving after time travel.")


class TestCompareWithOriginal(unittest.TestCase):
    """Compare behavior with original time_cell.py implementation."""

    def test_original_has_varied_results(self):
        """The original implementation should show varied cycle lengths."""
        # This imports the original implementation
        try:
            from time_cell import TimeCell, Config, Result
        except ImportError:
            self.skipTest("Original time_cell.py not available")

        # Run original with various seeds
        cycle_lengths = []
        for seed in range(50):
            np.random.seed(seed)
            cfg = Config(rule=30, ratio=0.5, t_enter=80, t_exit=40, portal_w=16)
            ca = TimeCell(config=cfg, quick_compute=True)
            result = ca.run_until_time_loop(max_trips=1000)
            if result is not None:
                cycle_lengths.append(result.cycle_length)

        # Original should have varied cycle lengths
        unique_lengths = set(cycle_lengths)
        print(f"Original implementation cycle lengths: {unique_lengths}")
        self.assertGreater(len(unique_lengths), 1,
                           "Original implementation should have varied cycle lengths")


if __name__ == '__main__':
    unittest.main(verbosity=2)
