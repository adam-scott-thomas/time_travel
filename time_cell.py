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

from typing import Tuple, Optional, Dict, List, Union
from copy import copy
from collections import namedtuple

import pygame
import numpy as np
import numpy.typing as npt

SCREEN: Optional[pygame.Surface] = None
WIDTH: int = 1280
HEIGHT: int = 720


def rect(color: Tuple[int, int, int], left: float, top: float, width: float, height: float) -> None:
    """Draw a rectangle on the global SCREEN surface."""
    py_rect = pygame.Rect(left, top, width, height)
    pygame.draw.rect(SCREEN, color, py_rect)

Result = namedtuple("Result", "cycle_start cycle_end cycle_length")
Config = namedtuple("Config", "rule ratio t_enter t_exit portal_w")

def rule_name_to_list(rule_name: int) -> npt.NDArray[np.int8]:
    """Convert Wolfram rule number to binary lookup table.

    Args:
        rule_name: Wolfram rule number (0-255)

    Returns:
        8-element numpy array representing the rule's binary encoding
        e.g. 110 -> [0,1,1,0,1,1,1,0]
    """
    rule_bin_str = np.binary_repr(rule_name, width=8)
    return np.array([int(x) for x in rule_bin_str], dtype=np.int8)

class TimeCell:
    """A 1D cellular automaton simulation with time travel portals.

    This class implements a cellular automaton that can send patterns backwards in time
    through "time portals", creating the possibility of temporal loops and paradoxes.

    Attributes:
        scl: Pixel scale for rendering
        num_cells: Number of cells in the automaton
        num_gens: Number of generations to simulate
        universe: 2D array representing the cellular automaton grid
        active_generations: List of currently active time steps
        quick_compute: If True, optimize for speed over visualization
        config: Configuration parameters
        rules: Binary lookup table for the Wolfram rule
        result: Time loop detection result (if found)
        t_enter/t_exit: Time portal entry and exit points
        x_enter/x_exit: Spatial portal entry and exit points
        w: Portal width
        history: Cache of patterns sent through portal
        trips: Number of time portal traversals
    """

    def __init__(self, config: Optional[Config] = None, quick_compute: bool = True, center: bool = False) -> None:
        """Initialize a time travel cellular automaton.

        Args:
            config: Configuration object with simulation parameters
            quick_compute: Only update earliest active row for speed
            center: Start with single center cell active (overrides config.ratio)
        """
        self.scl: int = 5  # Pixel scale for rendering
        self.num_cells: int = int(WIDTH / self.scl)
        self.num_gens: int = int(HEIGHT / self.scl)

        self.num_steps: int = 0
        self.active_generations: List[int] = []
        self.universe: Optional[npt.NDArray[np.int8]] = None
        self.quick_compute: bool = quick_compute
        self.result: Optional[Result] = None

        # Configuration
        self.config: Config = config or Config(rule=110, ratio=0.2, t_enter=80, t_exit=40, portal_w=32)
        self.rules: npt.NDArray[np.int8] = rule_name_to_list(self.config.rule)
        self.restart(self.config.ratio, center)

        # Time portal parameters
        self.t_enter: int = self.config.t_enter
        self.t_exit: int = self.config.t_exit
        self.x_enter: int = int(self.num_cells / 2 + 5)
        self.x_exit: int = int(self.num_cells / 2 + 5)
        self.w: int = self.config.portal_w
        self.history: Dict[Tuple[int, ...], int] = {}  # Pattern -> trip number
        self.trips: int = 0

    def restart(self, ratio: float, center: bool = False) -> None:
        """Reset the universe to initial conditions at time t = 0.

        Args:
            ratio: Probability that each cell starts active (0.0-1.0)
            center: If True, start with only center cell active
        """
        self.num_steps = 0
        self.universe = np.zeros(shape=(self.num_gens, self.num_cells), dtype=np.int8)
        self.active_generations = [0]

        if center:
            self.universe[0][self.num_cells // 2] = 1
        else:
            self.universe[0] = (np.random.rand(self.num_cells) < ratio).astype(np.int8)

    def generate(self) -> None:
        """Evolve all active generations forward by one time step.

        This is the main simulation loop that:
        1. Filters out generations that have reached maximum age
        2. Applies cellular automaton rules to evolve each active generation
        3. Checks for time portal activation and loop detection
        """
        self.num_steps += 1
        # Filter out generations that have reached maximum age
        self.active_generations = [t for t in self.active_generations if t < self.num_gens - 1]

        if self.quick_compute:
            # Performance optimization: only simulate the earliest generation
            # when we're just trying to detect time loops
            if self.active_generations:
                self.active_generations = [min(self.active_generations)]

        for i in range(len(self.active_generations)):
            t = self.active_generations[i]
            self.active_generations[i] += 1
            next_row = self.generate_row(t)
            self.universe[t + 1] = next_row
            self.check_row_for_portal_and_loops(t)

    def generate_row(self, t: int) -> npt.NDArray[np.int8]:
        """Apply cellular automaton rule to generate next row.

        Uses vectorized operations to efficiently compute the next generation
        of the cellular automaton using the configured Wolfram rule.

        Args:
            t: Time step to evolve from

        Returns:
            Next generation row as numpy array
        """
        # Extract left, center, right neighbors for vectorized rule application
        row_l = self.universe[t][0:-2]  # Left neighbors
        row_c = self.universe[t][1:-1]  # Center cells
        row_r = self.universe[t][2:]    # Right neighbors

        # Convert (left, center, right) triplets to indices for rule lookup
        row_code = 4 * row_l + 2 * row_c + 1 * row_r

        # Apply Wolfram rule via vectorized lookup
        result = self.rules[7 - row_code]

        # Update next row (edges remain 0)
        new_row = self.universe[t + 1]
        new_row[1:-1] = result
        return new_row

    def check_row_for_portal_and_loops(self, t: int) -> None:
        """Handle time portal activation and detect temporal loops.

        When the simulation reaches the portal entry time, this method:
        1. Copies a section of the current state back to an earlier time
        2. Checks if we've sent the same pattern before (loop detection)
        3. Records the pattern in our history for future loop detection

        Args:
            t: Current time step being processed
        """
        if t == self.t_enter:
            # Activate time portal: add exit time to active generations
            self.active_generations.append(self.t_exit)

            # Copy portal contents from entry point back to exit point
            portal_contents = copy(self.universe[t + 1][self.x_enter:self.x_enter + self.w])
            self.universe[self.t_exit][self.x_exit:self.x_exit + self.w] = portal_contents

            # Record this time travel event
            self.trips += 1
            portal_contents_tup = tuple(portal_contents)

            # Check if we've sent this exact pattern before (temporal loop detection)
            if portal_contents_tup in self.history:
                self.result = Result(
                    cycle_start=self.history[portal_contents_tup],
                    cycle_end=self.trips,
                    cycle_length=self.trips - self.history[portal_contents_tup],
                )

            # Update history with this pattern
            self.history[portal_contents_tup] = self.trips

    def run_until_time_loop(self, max_trips: Optional[int] = 400, render: bool = False) -> Optional[Result]:
        """Run the simulation until a temporal loop is detected.

        Continues evolving the cellular automaton and checking for time portals
        until either a stable temporal loop is found or the maximum trip limit is reached.

        Args:
            max_trips: Maximum number of portal traversals before giving up
            render: If True, display pygame visualization during simulation

        Returns:
            Result object with loop details if found, None if no loop detected
        """
        if render:
            pygame.init()
            global SCREEN
            SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

        self.quick_compute = True  # Optimize for speed
        while self.result is None:
            if max_trips is not None and self.trips > max_trips:
                return None
            if render:
                self.render()
            self.generate()

        return self.result

    def render(self) -> None:
        """Draw the cellular automaton and time portals using pygame.

        Renders all active generations of the CA along with visual indicators
        for the time portal entry (green) and exit (red) points.
        """
        for t in self.active_generations:
            self._render_row(t)

        red = (255, 0, 0)
        green = (0, 255, 0)
        scl = self.scl

        # Draw exit portal (red) - where patterns appear from the future
        rect(red, self.x_exit * scl, self.t_exit * scl, self.w * scl, scl / 5)

        # Draw entry portal (green) - where patterns are sent to the past
        rect(green, self.x_enter * scl, self.t_enter * scl, self.w * scl, scl / 5)

        pygame.display.flip()

    def _render_row(self, t: int) -> None:
        """Render a single row of the cellular automaton at time t.

        Args:
            t: Time step to render
        """
        scl = self.scl
        black = (0, 0, 0)
        white = (255, 255, 255)

        for i in range(self.num_cells):
            color = white if self.universe[t][i] == 1 else black
            rect(color, i * scl, t * scl, scl, scl)


def loop() -> None:
    """Run interactive simulation with pygame visualization.

    Displays the cellular automaton with time portals and allows user
    to watch temporal loop formation in real-time.
    """
    quick = False
    cfg = Config(rule=30, ratio=0.2, t_enter=80, t_exit=40, portal_w=32)
    ca = TimeCell(config=cfg, quick_compute=quick, center=True)

    while True:
        ca.render()
        ca.generate()

        if ca.result is not None:
            print("Time loop found!")
            print(ca.result)
            if quick:
                return

        # Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return

def several_loops() -> None:
    """Run sequential experiments with different Wolfram rules.

    Cycles through a curated set of interesting Wolfram rules,
    visualizing their temporal loop behavior with pygame.
    """
    rules = [30, 45, 73, 97, 110, 137, 161, 165, 169]
    ratios = [0.5]
    for rule in rules:
        for ratio in ratios:
            cfg = Config(rule=rule, ratio=ratio, t_enter=80, t_exit=40, portal_w=32)
            print(cfg)
            ca = TimeCell(config=cfg, quick_compute=False)

            done_count = 0
            for _ in range(1500):
                ca.render()
                ca.generate()

                if ca.result is not None:
                    done_count += 1
                    if done_count >= 200:
                        break

                # Handle user input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return


if __name__ == "__main__":
    pygame.init()
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

    several_loops()
