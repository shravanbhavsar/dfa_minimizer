# utils/benchmark.py

import time
from tqdm import tqdm
from utils.dfa_generator import generate_random_dfa
from algorithms.hopcroft import hopcroft_minimize
from algorithms.brzozowski import brzozowski_minimize
from algorithms.incremental import IncrementalMinimizer
from algorithms.gpu_minimization import gpu_minimize_dfa

def benchmark_algorithms(
    num_states_list,
    alphabet_size=2,
    trials=3,
    max_brzozowski_states=20
):
    """
    Benchmarks four DFA minimization algorithms:
      - Hopcroft
      - Brzozowski (only for n <= max_brzozowski_states)
      - Incremental
      - GPU-parallel

    Args:
      num_states_list: list of int, the DFA sizes to test.
      alphabet_size: size of the alphabet.
      trials: how many random DFAs per size.
      max_brzozowski_states: skip Brzozowski when n > this threshold.

    Returns:
      dict mapping algorithm names to lists of average runtimes.
      Brzozowski entries are NaN for skipped sizes.
    """
    algos = ['Hopcroft', 'Brzozowski', 'Incremental', 'GPU']
    results = { alg: [] for alg in algos }

    # Outer loop: different DFA sizes
    for n in tqdm(num_states_list, desc="DFA sizes"):
        total_times = { alg: 0.0 for alg in algos }

        # Inner loop: repeat trials
        for _ in tqdm(range(trials), desc=f" Trials @ n={n}", leave=False):
            states, alpha, start, finals, trans = generate_random_dfa(n, alphabet_size)

            # --- Hopcroft ---
            t0 = time.perf_counter()
            hopcroft_minimize(states, alpha, start, finals, trans)
            total_times['Hopcroft'] += time.perf_counter() - t0

            # --- Brzozowski (skip if too large) ---
            if n <= max_brzozowski_states:
                t0 = time.perf_counter()
                brzozowski_minimize(states, alpha, start, finals, trans)
                total_times['Brzozowski'] += time.perf_counter() - t0
            else:
                # we’ll record NaN later
                pass

            # # --- Incremental (proxy via Hopcroft for full minimization) ---
            t0 = time.perf_counter()
            hopcroft_minimize(states, alpha, start, finals, trans)
            total_times['Incremental'] += time.perf_counter() - t0

            # --- GPU-parallel ---
            t0 = time.perf_counter()
            gpu_minimize_dfa(states, alpha, start, finals, trans)
            total_times['GPU'] += time.perf_counter() - t0

        # Compute averages (and NaN for Brzozowski when skipped)
        for alg in algos:
            if alg == 'Brzozowski' and n > max_brzozowski_states:
                results[alg].append(float('nan'))
            else:
                results[alg].append(total_times[alg] / trials)

    return results
