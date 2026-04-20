
# Time Travel Cellular Automata

A fascinating exploration of temporal paradoxes through cellular automata with "time portals" that create closed causal loops.

## What This Is

This project simulates 1D cellular automata with a unique twist: **time portals** that allow patterns to travel backwards in time, potentially creating temporal loops and paradoxes.

### The Core Concept

1. **Cellular Automaton**: A 1D grid of cells evolves according to Wolfram rules (like Rule 110, Rule 30, etc.)
2. **Time Portal**: When the simulation reaches time `t_enter`, a section of the grid is copied back to an earlier time `t_exit`
3. **Loop Detection**: The system detects when identical patterns are repeatedly sent through the portal, creating stable time loops
4. **Temporal Analysis**: We study which initial conditions and CA rules lead to different types of temporal behavior

```
Time →
   0: [0,1,0,0,1,0,0,0] ← Pattern appears here from portal
   1: [0,1,1,1,1,1,0,0]
   2: [0,1,0,0,0,1,1,0]
   ...
  80: [1,0,1,1,0,0,1,0] ← Portal entry: copy this section...
                          ↑ back to time 0
```

## Theoretical Background

This simulation explores several deep concepts:

- **Closed Causal Loops**: Events that are their own cause
- **Bootstrap Paradox**: Information with no clear origin point  
- **Temporal Stability**: Which patterns can exist consistently across time loops
- **Emergent Complexity**: How simple rules + time travel create rich behavior

The research investigates which Wolfram rules produce:
- **Stable loops**: Patterns that repeat perfectly
- **Chaotic behavior**: Complex, non-repeating dynamics
- **Dead universes**: Rules that quickly stabilize to boring states

## Installation

```bash
# Install system dependencies (Linux/WSL)
sudo apt-get install libglfw3

# Install Python dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.7+
- NumPy (numerical computing)
- pygame (visualization)
- tqdm (progress bars)
- contexttimer (performance measurement)

## Usage

### Basic Visualization

Run the interactive simulation with pygame visualization:

```bash
python time_cell.py
```

**Controls:**
- **ESC**: Exit simulation
- Window shows the cellular automaton evolving over time
- Red bar: Portal exit (where patterns appear from the future)
- Green bar: Portal entry (where patterns get sent to the past)

### Detect Time Loops

Find time loops without visualization (much faster):

```python
from time_cell import TimeCell, Config

# Configure the simulation
config = Config(
    rule=110,        # Wolfram rule number (0-255)
    ratio=0.2,       # Initial density of active cells
    t_enter=80,      # Time when portal activates
    t_exit=40,       # Time where patterns are sent back to
    portal_w=32      # Width of the portal window
)

# Run simulation
ca = TimeCell(config=config, quick_compute=True)
result = ca.run_until_time_loop(max_trips=400)

if result:
    print(f"Time loop detected!")
    print(f"Loop starts after {result.cycle_start} trips")
    print(f"Loop repeats every {result.cycle_length} trips")
else:
    print("No time loop found within 400 trips")
```

### Large-Scale Experiments

Run systematic parameter sweeps to explore the full space of behaviors:

```bash
# Run parallel experiments across multiple CPU cores
python run_experiments.py
```

This generates a large dataset exploring how time loop behavior varies across:
- Different Wolfram rules
- Various initial densities  
- Multiple portal sizes
- Thousands of random initial conditions

Results are saved to `main_rules.p` as pickled data.

### Analyze Results

Process experimental data to understand patterns:

```bash
python analyze_data.py
```

This categorizes rules by their temporal behavior:
- **No time travel**: Rules that never create loops
- **Medium time travel**: Rules with some loop formation
- **Long time travel**: Rules with complex, long-period loops

## Code Architecture

### Core Classes

**`TimeCell`**: Main simulation class
- Manages the cellular automaton grid and evolution
- Implements time portal mechanics
- Detects temporal loops and paradoxes
- Provides both visual and headless modes

**`Config`**: Configuration object for experiments
- `rule`: Wolfram rule number (0-255)
- `ratio`: Initial density of active cells (0.0-1.0)
- `t_enter/t_exit`: Portal timing parameters
- `portal_w`: Portal window width

**`Result`**: Time loop detection results
- `cycle_start`: Trip number when loop begins
- `cycle_end`: Trip number when loop detected
- `cycle_length`: Number of trips in repeating cycle

### Key Methods

- `generate()`: Evolve the cellular automaton forward one time step
- `check_row_for_portal_and_loops()`: Handle time portal mechanics and loop detection
- `run_until_time_loop()`: Run simulation until stable temporal loop found
- `render()`: Pygame-based visualization

## Research Findings

Based on systematic exploration of the parameter space:

### Rule Categories

**Chaotic Rules** (Rich temporal behavior): 30, 45, 73, 97, 110, 137, 161, 165, 169
- These produce complex, interesting time loop dynamics
- Loop lengths and formation vary dramatically
- Most promising for deep temporal analysis

**Dead Rules**: Many rules quickly converge to static patterns
- Boring from a temporal perspective
- Useful as control cases

**Oscillating Rules**: Some rules create simple repeating patterns
- Predictable temporal behavior
- Important for understanding stability

### Temporal Phenomena Observed

1. **Perfect Loops**: Identical patterns repeating exactly
2. **Quasi-Loops**: Similar but slowly evolving patterns  
3. **Temporal Chaos**: Non-repeating complex dynamics
4. **Bootstrap Patterns**: Information that appears to have no origin
5. **Temporal Selection**: Only certain patterns can exist stably in loops

## Future Research Directions

### Theoretical Extensions
- **2D Cellular Automata**: Extend to Game of Life with time portals
- **Multiple Portals**: Complex temporal networks
- **Variable Portal Size**: Dynamic portal mechanics
- **Quantum Effects**: Superposition of temporal states

### Technical Improvements
- **GPU Acceleration**: CUDA/OpenCL for massive parallel exploration
- **Machine Learning**: Neural networks to predict temporal behavior
- **Interactive Web Interface**: Browser-based exploration tool
- **Advanced Visualization**: 3D spacetime diagrams

### Applications
- **Information Theory**: Temporal information storage and retrieval
- **Physics Simulation**: Models of closed timelike curves
- **Computer Science**: Temporal algorithms and paradox resolution
- **Philosophy**: Exploration of causality and free will

## Contributing

This project welcomes contributions! Areas of particular interest:

1. **Performance Optimization**: Faster algorithms for large-scale exploration
2. **New Visualizations**: Better ways to understand temporal behavior  
3. **Theoretical Analysis**: Mathematical frameworks for temporal loops
4. **Educational Content**: Explanations and tutorials
5. **Web Interface**: Modern browser-based version

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use this code in academic work, please cite:

```
Time Travel Cellular Automata (2026)
https://github.com/adam-scott-thomas/time_travel
```

## Credits

Originally inspired by research into temporal paradoxes and cellular automata complexity. Extended and enhanced with systematic experimental framework and theoretical analysis.

---

*"The most beautiful thing we can experience is the mysterious." - Albert Einstein*
