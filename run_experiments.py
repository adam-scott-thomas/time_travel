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

import pickle
from typing import Tuple, Optional, Union, Iterator, Callable, Any
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from time_cell import TimeCell, Config, Result


def single_run(cfg: Config) -> Tuple[Config, Optional[Result]]:
    """Run a single time travel simulation with given configuration.

    Args:
        cfg: Configuration object with simulation parameters

    Returns:
        Tuple of (configuration, result) where result is None if no loop found
    """
    ca = TimeCell(config=cfg, quick_compute=True)
    res = ca.run_until_time_loop()
    return cfg, res

def count_pickles(f_name: str) -> int:
    """Count the number of objects stored in a pickle file.

    Args:
        f_name: Path to the pickle file

    Returns:
        Number of pickled objects in the file, 0 if file doesn't exist
    """
    count = 0
    try:
        with open(f_name, "rb") as f:
            while True:
                try:
                    pickle.load(f)
                    count += 1
                except EOFError:
                    return count
    except IOError:  # Handle case where file doesn't exist yet
        return 0

def run_job_server(
    func: Callable[[Config], Tuple[Config, Optional[Result]]],
    experiments: Union[list[Config], Iterator[Config]],
    save_file: str,
    resume: bool = True,
    num_experiments: Optional[int] = None,
    n_cores: Optional[int] = None
) -> None:
    """Run parallel experiments using multiprocessing with progress tracking and resume capability.

    This function provides a robust framework for running large-scale parameter sweeps:
    - Parallel execution across multiple CPU cores
    - Progress bar visualization with tqdm
    - Automatic resume capability for interrupted runs
    - Results saved incrementally to pickle file

    Args:
        func: Function to call for each experiment (typically single_run)
        experiments: List or generator of experiment configurations
        save_file: Path where pickled results will be saved
        resume: If True, resume from previously saved results
        num_experiments: Total number of experiments (required for generators)
        n_cores: Number of CPU cores to use (default: all available)
    """
    if n_cores is None:
        n_cores = cpu_count()

    if hasattr(experiments, "__len__"):
        num_experiments = len(experiments)

    # Resume from previous run if requested
    start_index = 0
    if resume:
        start_index = count_pickles(save_file)
        if start_index != 0:
            print(f"Resuming from index: {start_index}")
        # Skip already completed experiments
        for _ in range(start_index):
            next(experiments)

    with Pool(n_cores) as p:
        for result in tqdm(
            p.imap(single_run, experiments),
            total=num_experiments,
            initial=start_index
        ):
            # Save results incrementally to avoid memory buildup
            with open(save_file, "ab") as f:
                pickle.dump(result, f)


if __name__ == '__main__':
    # build experiments
    rules = [30, 45, 73, 97, 110, 137, 161, 165, 169]
    ratios = [.1, .5, .9]
    widths = [32, 33, 34, 35, 36, 37]
    reps = 1500*12

    # generate all the experiments to run with a cartesian product generator
    # for very large lists, generators are way faster
    experiments_gen = (Config(rule=rule, ratio=ratio, t_enter=80, t_exit=40, portal_w=width)
                for _ in range(reps)  # reps as outermost loop to spread out everything
                for rule in rules
                for ratio in ratios
                for width in widths)

    num_experiments = len(rules) * len(ratios) * len(widths) * reps

    # run!
    run_job_server(
        single_run,
        experiments_gen,
        "main_rules.p",
        num_experiments=num_experiments,
        resume=True)
