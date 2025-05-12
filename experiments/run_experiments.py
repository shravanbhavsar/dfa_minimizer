# experiments/run_experiments.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.benchmark import benchmark_algorithms

if __name__ == '__main__':
    # sizes = [100, 500, 1000]
    sizes = list(range(50, 1001, 50))
    results = benchmark_algorithms(sizes, alphabet_size=2, trials=3)

    print(results)

    for alg in tqdm(results.keys(), desc="Plotting"):
        plt.plot(sizes, results[alg], label=alg)

    plt.xlabel('Number of States')
    plt.ylabel('Average Runtime (s)')
    plt.title('DFA Minimization Algorithm Performance')
    plt.legend()
    plt.savefig('performance.png')
    print("Done! Plot saved to performance.png")
