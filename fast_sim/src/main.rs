//! Fast Time Travel Paradox Simulation
//!
//! Optimizations over Python version:
//! 1. No array allocations in hot loop - evolve row in place
//! 2. FxHash for fast hashing of portal states
//! 3. Bit-packed portal state for small widths
//! 4. Rayon for parallel execution

use rand::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use indicatif::{ProgressBar, ProgressStyle};

/// Configuration for a simulation run
#[derive(Clone, Copy)]
struct SimConfig {
    rule: u8,
    init_density: f64,
    t_enter: usize,
    t_exit: usize,
    portal_width: usize,
    num_cells: usize,
    num_generations: usize,
    portal_offset: i32,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            rule: 110,
            init_density: 0.5,
            t_enter: 80,
            t_exit: 40,
            portal_width: 32,
            num_cells: 256,
            num_generations: 144,
            portal_offset: 5,
        }
    }
}

/// Result of a simulation run
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SimResult {
    rule: u8,
    width: usize,
    density: f64,
    distance: usize,  // Time travel distance (t_enter - t_exit)
    seed: u64,
    found_loop: bool,
    cycle_length: usize,
    cycle_start: usize,
    total_trips: usize,
    is_perfect: bool,
}

/// Convert Wolfram rule number to lookup table
fn rule_to_lookup(rule: u8) -> [u8; 8] {
    let mut lookup = [0u8; 8];
    for i in 0..8 {
        lookup[7 - i] = (rule >> i) & 1;
    }
    lookup
}

/// Fast time travel simulator
struct TimeTravelSimulator {
    config: SimConfig,
    rule_lookup: [u8; 8],
    universe: Vec<Vec<u8>>,
    portal_x: usize,
    history: FxHashMap<Vec<u8>, usize>,
    trips: usize,
    current_gen: usize,
}

impl TimeTravelSimulator {
    fn new(config: SimConfig) -> Self {
        let rule_lookup = rule_to_lookup(config.rule);
        let universe = vec![vec![0u8; config.num_cells]; config.num_generations];
        let portal_x = ((config.num_cells as i32 / 2) + config.portal_offset) as usize;

        TimeTravelSimulator {
            config,
            rule_lookup,
            universe,
            portal_x,
            history: FxHashMap::default(),
            trips: 0,
            current_gen: 0,
        }
    }

    fn reset(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        // Clear universe
        for row in &mut self.universe {
            row.fill(0);
        }

        // Initialize first row
        for cell in &mut self.universe[0] {
            *cell = if rng.gen::<f64>() < self.config.init_density { 1 } else { 0 };
        }

        self.history.clear();
        self.trips = 0;
        self.current_gen = 0;
    }

    /// Evolve a single row - optimized to avoid allocations
    #[inline]
    fn evolve_row(&mut self, t: usize) {
        let cfg = &self.config;
        let n = cfg.num_cells;

        // Evolve each cell based on neighbors (with wrapping)
        for i in 0..n {
            let left = self.universe[t][(i + n - 1) % n];
            let center = self.universe[t][i];
            let right = self.universe[t][(i + 1) % n];

            // Pattern index: 111=7, 110=6, etc.
            let idx = 4 * left + 2 * center + right;
            self.universe[t + 1][i] = self.rule_lookup[7 - idx as usize];
        }
    }

    /// Check for portal entry and handle time travel
    /// Returns: (found_loop, time_traveled)
    fn check_portal(&mut self) -> (bool, bool) {
        let cfg = &self.config;

        if self.current_gen != cfg.t_enter {
            return (false, false);
        }

        // Extract portal contents from t_enter + 1
        let portal_end = self.portal_x + cfg.portal_width;
        let portal_contents: Vec<u8> = self.universe[self.current_gen + 1][self.portal_x..portal_end].to_vec();

        self.trips += 1;

        // Check for loop
        if let Some(&first_trip) = self.history.get(&portal_contents) {
            // Found a loop!
            return (true, true);
        }

        // Record this pattern
        self.history.insert(portal_contents.clone(), self.trips);

        // Send pattern back in time
        self.universe[cfg.t_exit][self.portal_x..portal_end].copy_from_slice(&portal_contents);

        // Reset to exit point
        self.current_gen = cfg.t_exit;

        (false, true)
    }

    /// Run simulation until loop found or max_trips reached
    fn run(&mut self, max_trips: usize, seed: u64) -> SimResult {
        self.reset(seed);

        let mut found_loop = false;
        let mut cycle_start = 0;
        let mut cycle_length = 0;

        while self.trips < max_trips {
            // Evolve current generation
            if self.current_gen < self.config.num_generations - 1 {
                self.evolve_row(self.current_gen);
            }

            // Check portal
            let (loop_found, time_traveled) = self.check_portal();

            if loop_found {
                // Get cycle info
                let portal_end = self.portal_x + self.config.portal_width;
                let portal_contents: Vec<u8> = self.universe[self.current_gen + 1][self.portal_x..portal_end].to_vec();

                cycle_start = *self.history.get(&portal_contents).unwrap_or(&0);
                cycle_length = self.trips - cycle_start;
                found_loop = true;
                break;
            }

            // Only increment if we didn't time travel
            if !time_traveled {
                self.current_gen += 1;
            }
        }

        SimResult {
            rule: self.config.rule,
            width: self.config.portal_width,
            density: self.config.init_density,
            distance: self.config.t_enter - self.config.t_exit,
            seed,
            found_loop,
            cycle_length,
            cycle_start,
            total_trips: self.trips,
            is_perfect: found_loop && cycle_length == 1,
        }
    }
}

/// Run a batch of experiments varying time travel distance
fn run_distance_experiment(
    rules: &[u8],
    widths: &[usize],
    distances: &[usize],
    reps: usize,
    max_trips: usize,
) -> Vec<SimResult> {
    let total = rules.len() * widths.len() * distances.len() * reps;
    println!("Running {} experiments...", total);

    let pb = ProgressBar::new(total as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    // Generate all configs with varying time travel distances
    let configs: Vec<(SimConfig, u64)> = rules.iter().flat_map(|&rule| {
        widths.iter().flat_map(move |&width| {
            distances.iter().flat_map(move |&distance| {
                (0..reps).map(move |rep| {
                    // t_exit at 20, t_enter = t_exit + distance
                    // num_generations needs to be > t_enter + some buffer
                    let t_exit = 20;
                    let t_enter = t_exit + distance;
                    let num_gens = t_enter + 50;  // Buffer after portal entry

                    let config = SimConfig {
                        rule,
                        portal_width: width,
                        init_density: 0.5,
                        t_enter,
                        t_exit,
                        num_generations: num_gens,
                        ..Default::default()
                    };
                    // Deterministic seed
                    let seed = ((rule as u64) << 48) | ((width as u64) << 32) |
                               ((distance as u64) << 16) | (rep as u64);
                    (config, seed)
                })
            })
        })
    }).collect();

    // Run in parallel
    let results: Vec<SimResult> = configs
        .par_iter()
        .map(|(config, seed)| {
            let mut sim = TimeTravelSimulator::new(*config);
            let result = sim.run(max_trips, *seed);
            pb.inc(1);
            result
        })
        .collect();

    pb.finish_with_message("Done!");
    results
}

/// Aggregate results and print summary
fn summarize(results: &[SimResult]) {
    println!("\n{}", "=".repeat(60));
    println!("SUMMARY");
    println!("{}", "=".repeat(60));

    let total = results.len();
    let loops_found = results.iter().filter(|r| r.found_loop).count();
    let perfect_loops = results.iter().filter(|r| r.is_perfect).count();
    let oscillating = loops_found - perfect_loops;

    println!("\nOverall: {} experiments", total);
    println!("  Loops found: {:.2}%", 100.0 * loops_found as f64 / total as f64);
    println!("  Perfect loops: {:.2}%", 100.0 * perfect_loops as f64 / total as f64);
    println!("  Oscillating loops: {:.2}%", 100.0 * oscillating as f64 / total as f64);

    // Mean cycle length
    let cycles: Vec<usize> = results.iter()
        .filter(|r| r.found_loop)
        .map(|r| r.cycle_length)
        .collect();

    if !cycles.is_empty() {
        let mean_cycle = cycles.iter().sum::<usize>() as f64 / cycles.len() as f64;
        let max_cycle = cycles.iter().max().unwrap_or(&0);
        println!("  Mean cycle length: {:.2}", mean_cycle);
        println!("  Max cycle length: {}", max_cycle);
    }

    // By rule
    println!("\nBy Rule:");
    let mut by_rule: HashMap<u8, Vec<&SimResult>> = HashMap::new();
    for r in results {
        by_rule.entry(r.rule).or_default().push(r);
    }
    let mut rules: Vec<_> = by_rule.keys().collect();
    rules.sort();

    for rule in rules {
        let data = &by_rule[rule];
        let loops = data.iter().filter(|r| r.found_loop).count();
        let perfect = data.iter().filter(|r| r.is_perfect).count();
        let cycles: Vec<_> = data.iter().filter(|r| r.found_loop).map(|r| r.cycle_length).collect();
        let mean_cycle = if cycles.is_empty() { 0.0 } else {
            cycles.iter().sum::<usize>() as f64 / cycles.len() as f64
        };

        println!("  Rule {:3}: loop={:.0}%, perfect={:.1}%, mean_cycle={:.1}",
            rule,
            100.0 * loops as f64 / data.len() as f64,
            100.0 * perfect as f64 / data.len() as f64,
            mean_cycle
        );
    }

    // By width
    println!("\nBy Width:");
    let mut by_width: HashMap<usize, Vec<&SimResult>> = HashMap::new();
    for r in results {
        by_width.entry(r.width).or_default().push(r);
    }
    let mut widths: Vec<_> = by_width.keys().collect();
    widths.sort();

    for width in widths {
        let data = &by_width[width];
        let loops = data.iter().filter(|r| r.found_loop).count();
        let cycles: Vec<_> = data.iter().filter(|r| r.found_loop).map(|r| r.cycle_length).collect();
        let mean_cycle = if cycles.is_empty() { 0.0 } else {
            cycles.iter().sum::<usize>() as f64 / cycles.len() as f64
        };
        let pre_cycles: Vec<_> = data.iter().filter(|r| r.found_loop).map(|r| r.cycle_start - 1).collect();
        let mean_pre = if pre_cycles.is_empty() { 0.0 } else {
            pre_cycles.iter().sum::<usize>() as f64 / pre_cycles.len() as f64
        };

        let state_space = 2u64.pow(*width as u32);
        let expected = (std::f64::consts::PI * state_space as f64 / 2.0).sqrt();

        println!("  Width {:2} (2^{:2}={:>8}): loop={:.0}%, mean_cycle={:.1}, mean_pre={:.1} (exp: {:.0})",
            width, width, state_space,
            100.0 * loops as f64 / data.len() as f64,
            mean_cycle, mean_pre, expected
        );
    }
}

fn main() {
    use std::time::Instant;

    println!("=== Time Travel Distance Experiment (High Resolution) ===\n");

    // Test how behavior changes with time travel distance
    // Every step from 5-100 to detect any periodic patterns

    let rules: Vec<u8> = vec![30, 110];  // Chaotic rules separately
    let widths: Vec<usize> = vec![16];   // Fix width to reduce variables
    let distances: Vec<usize> = (5..=100).collect();  // Every step from 5-100
    let reps = 2500;  // 5x more samples for statistical significance
    let max_trips = 100000;

    println!("Configuration:");
    println!("  Rules: {:?}", rules);
    println!("  Width: {:?}", widths);
    println!("  Distances: 5-100 (every step, {} values)", distances.len());
    println!("  Reps per config: {}", reps);

    let total = rules.len() * widths.len() * distances.len() * reps;
    println!("  Total experiments: {}", total);
    println!();

    let start = Instant::now();
    let results = run_distance_experiment(&rules, &widths, &distances, reps, max_trips);
    let elapsed = start.elapsed();

    println!("\nCompleted in {:.2}s ({:.1} experiments/sec)",
        elapsed.as_secs_f64(),
        results.len() as f64 / elapsed.as_secs_f64()
    );

    // Summary by distance for each rule separately
    for rule in &rules {
        println!("\n{}", "=".repeat(70));
        println!("RULE {} - RESULTS BY TIME TRAVEL DISTANCE", rule);
        println!("{}", "=".repeat(70));
        println!("\n{:>8} {:>10} {:>10} {:>10} {:>8}", "Dist", "MeanCycle", "MaxCycle", "Perfect%", "N");
        println!("{}", "-".repeat(70));

        for dist in &distances {
            let data: Vec<_> = results.iter()
                .filter(|r| r.rule == *rule && r.distance == *dist)
                .collect();

            if data.is_empty() { continue; }

            let cycles: Vec<_> = data.iter().filter(|r| r.found_loop).map(|r| r.cycle_length).collect();
            let perfect = data.iter().filter(|r| r.is_perfect).count();
            let mean_cycle = if cycles.is_empty() { 0.0 } else {
                cycles.iter().sum::<usize>() as f64 / cycles.len() as f64
            };
            let max_cycle = cycles.iter().max().unwrap_or(&0);

            println!("{:>8} {:>10.1} {:>10} {:>9.1}% {:>8}",
                dist, mean_cycle, max_cycle,
                100.0 * perfect as f64 / data.len() as f64,
                data.len()
            );
        }
    }

    // Save results
    std::fs::create_dir_all("../results").ok();
    let file = File::create("../results/distance_highres.json").expect("Failed to create results file");
    serde_json::to_writer_pretty(file, &results).expect("Failed to write results");
    println!("\nSaved detailed results to results/distance_highres.json");
}
