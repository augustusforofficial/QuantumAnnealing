#!/usr/bin/env python3
"""Plot amplitudes/probabilities produced by the cuBLAS quantum annealing CUDA executable.

This script runs the cuBLAS_new_template executable, parses the probability lines,
and draws a bar chart using matplotlib. The horizontal axis is the state index
and the vertical axis is the probability.

Usage::

    python3 plot_cuBLAS_annealing.py [--output plot.png] [--threshold 0.01] [--top 20]

The executable is expected to be in the 'bin' directory as 'cuBLAS_new_template'.

Options:
  --output, -o:   Output file path (default: cuBLAS_plot.png)
  --threshold:    Only show states with probability >= threshold
  --top N:        Show only top N states with highest probabilities
"""

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import numpy as np


def read_probabilities_from_executable(exe_path: str) -> np.ndarray:
    """Run the given executable and extract probability values.

    The executable is expected to write lines containing ``<index> : <prob>`` to
    stdout. Only the probability values are returned as a NumPy array ordered
    by index.
    """
    proc = subprocess.run([exe_path], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"executable failed with code {proc.returncode}: {proc.stderr}")

    probs = []
    for line in proc.stdout.splitlines():
        # look for the lines "i : p" at the end of print_state
        if ":" not in line:
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0].strip())
            p = float(parts[1].strip())
        except ValueError:
            # skip lines that are not formatted as expected
            continue
        # ensure list is large enough
        if idx >= len(probs):
            probs.extend([0.0] * (idx + 1 - len(probs)))
        probs[idx] = p
    return np.array(probs)


def plot_probabilities(probs: np.ndarray, output: str = None, threshold: float = None, top_n: int = None):
    """Draw a bar chart of the probabilities.

    ``probs`` is a 1D array containing probabilities for states 0..len(probs)-1.

    Args:
        probs: Array of probabilities
        output: Output file path. If None, save to 'cuBLAS_plot.png'
        threshold: Only show states with probability >= threshold
        top_n: Show only top N states with highest probabilities
    """
    states = np.arange(probs.size)

    # Filter by threshold or top_n
    if threshold is not None:
        mask = probs >= threshold
        filtered_states = states[mask]
        filtered_probs = probs[mask]
    elif top_n is not None:
        top_indices = np.argsort(probs)[-top_n:]
        filtered_states = states[top_indices]
        filtered_probs = probs[top_indices]
    else:
        filtered_states = states
        filtered_probs = probs

    # Sort by probability for better visualization
    sorted_indices = np.argsort(filtered_probs)
    sorted_states = filtered_states[sorted_indices]
    sorted_probs = filtered_probs[sorted_indices]

    # Use square figure
    plt.figure(figsize=(8, 8))
    plt.bar(range(len(sorted_probs)), sorted_probs, width=0.8)
    plt.xlabel("State index")
    plt.ylabel("Probability")
    plt.title("cuBLAS Quantum Annealing State Probabilities")
    plt.xticks(range(len(sorted_probs)), sorted_states, rotation=45, fontsize=10)
    plt.tight_layout()

    if output:
        plt.savefig(output)
        print(f"Saved plot to {output}")
    else:
        output_file = "cuBLAS_plot.png"
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot probabilities from the cuBLAS quantum annealing executable.")
    parser.add_argument("--output", "-o", help="Output file path (default: cuBLAS_plot.png).")
    parser.add_argument("--threshold", type=float, help="Only show states with probability >= threshold.")
    parser.add_argument("--top", type=int, help="Show only top N states with highest probabilities.")
    args = parser.parse_args()

    exe_path = os.path.join("bin", "cuBLAS_new_template")
    if not os.path.isfile(exe_path):
        print(f"Executable not found: {exe_path}", file=sys.stderr)
        sys.exit(1)

    probs = read_probabilities_from_executable(exe_path)
    if probs.size == 0:
        print("No probabilities were found in the executable output.", file=sys.stderr)
        sys.exit(1)
    plot_probabilities(probs, args.output, threshold=args.threshold, top_n=args.top)


if __name__ == "__main__":
    main()