#!/usr/bin/env python3
"""Fast plotting for quantum annealing results using binary I/O.

The C executable should write probabilities directly to a binary file as double array:

    FILE *fp = fopen("result.bin", "wb");
    fwrite(prob, sizeof(double), Nums, fp);
    fclose(fp);

This script reads that binary file directly and plots either top-N states
or states above a threshold.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_prob_binary(filepath: str) -> np.ndarray:
    """Read double probabilities from binary file."""
    probs = np.fromfile(filepath, dtype=np.float64)
    if probs.size == 0:
        raise RuntimeError(f"No data found in {filepath}")
    return probs


def plot_probabilities(probs: np.ndarray, output: str, threshold: float = None, top_n: int = None):
    states = np.arange(probs.size)

    if top_n is not None:
        top_n = min(top_n, probs.size)
        idx = np.argpartition(probs, -top_n)[-top_n:]
        filtered_states = states[idx]
        filtered_probs = probs[idx]
    elif threshold is not None:
        mask = probs >= threshold
        filtered_states = states[mask]
        filtered_probs = probs[mask]
    else:
        filtered_states = states
        filtered_probs = probs

    order = np.argsort(filtered_probs)[::-1]
    filtered_states = filtered_states[order]
    filtered_probs = filtered_probs[order]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(filtered_probs)), filtered_probs)
    plt.xticks(range(len(filtered_states)), filtered_states, rotation=45)
    plt.xlabel("State index")
    plt.ylabel("Probability")
    plt.title("Quantum Annealing State Probabilities")
    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot annealing result from binary probability vector")
    parser.add_argument("binary_file", help="Binary file written by C executable")
    parser.add_argument("--output", "-o", default="plot.png")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    probs = read_prob_binary(args.binary_file)
    plot_probabilities(probs, args.output, threshold=args.threshold, top_n=args.top)


if __name__ == "__main__":
    main()
