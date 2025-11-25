# Time Travel Paradoxes in Cellular Automata: A Statistical Study

## TL;DR

I simulated "time travel" in 1D cellular automata over **160,000 experiments** to see what kinds of paradoxes emerge. The key findings:

- **34% achieve perfect loops** - the same pattern gets sent back forever (stable time loop)
- **66% are oscillating loops** - patterns cycle through multiple states before repeating
- **Mean cycle length: 6.7 trips** - most oscillations are short, but some reach 300+ trips
- **No infinite paradoxes** - every simulation eventually converges to a repeating pattern
- **Behavior varies dramatically by rule** - Rule 30 (chaotic) averages 18-trip cycles, while Rule 90 always produces perfect loops

The universe *eventually* achieves self-consistency, but often only after "trying out" different patterns!

---

## The Experiment

### What Are Cellular Automata?

1D Cellular Automata (CA) are simple computational systems where a row of cells evolves over time based on local rules. Each cell looks at itself and its neighbors, then decides its next state. Despite their simplicity, CA can produce remarkably complex behavior.

The most famous is **Rule 110**, which is actually Turing-complete - meaning it can compute anything a regular computer can!

### Simulating Time Travel

I created a "time portal" in the cellular automata universe:

1. **Start**: Initialize a random row of cells
2. **Evolve**: Run the CA forward in time
3. **Portal Entry**: At time t=80, capture cells in a specific region
4. **Portal Exit**: Insert those cells at time t=40 (in the past!)
5. **Re-evolve**: Continue from t=40 with the modified state
6. **Repeat**: Check if the portal captures the same pattern as before

The key question: **How many trips through the time machine before the pattern repeats?**

### The Paradox Tropes I Expected

Based on sci-fi movies, I expected to find:

- **Perfect Loops**: Pattern A goes back in time, causes exactly pattern A to be sent back (cycle=1)
- **Oscillating Loops**: Pattern A leads to B, B leads to C, C leads back to A (cycle>1)
- **Growing Chaos**: Each trip through the portal creates increasingly different results
- **Convergence After Struggle**: Initial chaos that eventually settles into a pattern

---

## The Results

### Finding 1: Both Loop Types Exist!

After running **160,000 simulations** across:
- 8 different Wolfram rules (30, 45, 73, 90, 110, 150, 169, 182)
- Portal widths from 6 to 20 cells
- Initial densities from 10% to 90%
- 500 random seeds per configuration

| Metric | Value |
|--------|-------|
| Total experiments | 160,000 |
| Loops found | 100% |
| Perfect loops (cycle=1) | 34.1% |
| Oscillating loops (cycle>1) | 65.9% |
| Mean cycle length | 6.68 trips |
| Max cycle length | 308 trips |
| Mean pre-cycle length | 4.08 trips |
| Max pre-cycle length | 296 trips |

### Finding 2: Behavior Depends Strongly on the CA Rule

| Rule | Perfect Loops | Mean Cycle | Max Cycle |
|------|--------------|------------|-----------|
| 30 (chaotic) | 8.3% | 18.0 | 308 |
| 45 | 18.7% | 5.7 | 77 |
| 73 | 11.7% | 5.7 | 146 |
| **90 (simple)** | **100%** | **1.0** | **1** |
| 110 (Turing-complete) | 26.5% | 6.3 | 82 |
| 150 | 16.8% | 13.1 | 34 |
| **169** | **73.5%** | **1.5** | **18** |
| 182 | 17.3% | 2.2 | 14 |

**Rule 90** (XOR rule) *always* produces perfect loops - its linear structure guarantees self-consistency.

**Rule 30** (famous for generating randomness) produces the longest cycles and most chaotic behavior, with only 8% achieving immediate stability.

### Finding 3: Portal Width Matters

Larger portals = larger state space = longer cycles before finding a repeated pattern.

**Chaotic rules only (30, 110)** - shows the clearest trend:

| Width | State Space | Mean Cycle | Max Cycle | Perfect Loop % |
|-------|-------------|------------|-----------|----------------|
| 6 | 64 | 3.6 | 21 | 26.6% |
| 8 | 256 | 5.0 | 33 | 20.7% |
| 10 | 1,024 | 6.6 | 51 | 18.3% |
| 12 | 4,096 | 8.8 | 96 | 15.9% |
| 14 | 16,384 | 11.4 | 96 | 15.0% |
| 16 | 65,536 | 15.4 | 155 | 14.0% |
| 18 | 262,144 | 19.7 | 202 | 13.6% |
| 20 | 1,048,576 | 25.7 | 268 | 13.6% |
| 22 | 4,194,304 | 35.2 | 395 | 12.8% |
| 24 | 16,777,216 | 47.2 | 489 | 13.5% |
| 26 | 67,108,864 | 65.2 | 636 | 14.0% |
| 28 | 268,435,456 | 88.4 | 1,138 | 13.4% |
| 30 | 1,073,741,824 | 123.6 | 1,441 | 12.5% |
| 32 | 4,294,967,296 | 169.4 | 1,965 | 11.5% |

Mean cycle length grows steadily with portal width, while perfect loop rate decreases. This makes sense: larger state spaces have more possible patterns to explore before finding a repeat.

Note: When aggregated across ALL rules (including Rule 90 which always produces cycle=1), these trends are diluted. The table above isolates the chaotic rules where oscillating loops are common.

### Finding 4: Initial Density Has Minimal Effect

| Density | Perfect Loop Rate | Mean Cycle |
|---------|-------------------|------------|
| 0.1 | 35.7% | 6.8 |
| 0.3 | 31.8% | 6.6 |
| 0.5 | 31.8% | 6.6 |
| 0.7 | 33.7% | 6.7 |
| 0.9 | 37.5% | 6.8 |

The extreme densities (0.1 and 0.9) slightly favor perfect loops, but the effect is minor.

---

## Why Does This Happen?

### The Mathematical View

Consider time travel as a function:
- Let `f` be the CA evolution from t_exit to t_enter
- Let `P` extract the portal region
- A time loop asks: find the smallest `k` where `(P o f)^k(x) = (P o f)^j(x)` for some `j < k`

This is asking for an **eventual cycle** in a finite state machine. Since there are only `2^width` possible portal states, the system *must* eventually repeat - the question is how long it takes.

### Why Different Rules Behave Differently

**Rule 90** (XOR) is *linear* over GF(2) - the state after k steps is a linear function of the initial state. This linear structure means the time travel dynamics form a very constrained system with many fixed points.

**Rule 30** is highly *nonlinear* and chaotic. The state space is explored more thoroughly before landing on a cycle, leading to longer pre-cycle and cycle lengths.

### The Novikov Connection

This relates to the **Novikov self-consistency principle** from physics: if time travel exists, the universe only allows self-consistent histories. Our simulations show that:

1. Self-consistency is *always* achieved (all simulations find loops)
2. But it may take multiple "iterations" to find the consistent state
3. The number of iterations depends on the physics (CA rule)

---

## Performance: Python vs Rust

I implemented the simulation in both Python and Rust:

| Implementation | Time for 160k experiments | Speed |
|---------------|---------------------------|-------|
| Python (multiprocessing) | 119.2 seconds | 1,342 exp/sec |
| Rust (rayon) | 5.5 seconds | 29,181 exp/sec |
| **Speedup** | | **~22x** |

The Rust version avoids Python overhead and numpy array allocations in the hot loop. For large-scale experiments, the Rust version is strongly recommended.

---

## Try It Yourself

### Installation

```bash
pip install numpy matplotlib tqdm
```

### Quick Test

```python
from simulation import SimConfig, TimeTravelSimulator

config = SimConfig(
    rule=110,
    portal_width=16,
    init_density=0.5,
    t_enter=80,
    t_exit=40,
)

sim = TimeTravelSimulator(config)
result = sim.run(max_trips=1000, seed=42)

print(f"Loop found: {result.found_loop}")
print(f"Cycle length: {result.cycle_length}")  # How many trips in the cycle
print(f"Pre-cycle: {result.pre_cycle_length}")  # Trips before cycle started
print(f"Perfect loop: {result.is_perfect_loop}")  # True if cycle_length=1
```

### Run Experiments

```bash
# Focused experiment (Python)
python focused_experiment.py

# With profiling
python focused_experiment.py profile

# Rust version (much faster!)
cd fast_sim
cargo build --release
./target/release/time_travel_sim
```

---

## Conclusion

Time travel in cellular automata is neither trivially stable nor hopelessly chaotic. The key findings:

1. **Loops always exist** - the finite state space guarantees eventual repetition
2. **Most loops oscillate** - 66% of simulations cycle through multiple states
3. **Rule matters enormously** - chaotic rules produce longer cycles
4. **Larger portals take longer** - but growth is sublinear relative to state space

The sci-fi implications:
- Grandfather paradoxes are impossible (the universe finds consistency)
- But you might experience different "timeline iterations" before stability
- Simple physics = immediate stability; complex physics = longer struggle

---

## Project Structure

```
time_travel/
    simulation.py           # Core CA + time travel simulation (Python)
    test_simulation.py      # Unit tests
    focused_experiment.py   # Main experiment runner
    fast_sim/               # Rust implementation (22x faster)
        src/main.rs
        Cargo.toml
    results/                # Experiment data (JSON)
        focused_results.json
        rust_results.json
```

---

## Future Work

- Investigate 2D cellular automata with time travel
- Explore stochastic CA rules (non-deterministic)
- Study time travel with multiple portals
- Analyze the structure of cycle attractors
- Compare with quantum cellular automata predictions
