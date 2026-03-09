#!/usr/bin/env python3
"""Plot amplitudes/probabilities produced by a quantum annealing C executable.

The C program (like ``new_templete.c``) prints complex amplitudes and then
lines of the form

    0 : 0.015625
    1 : 0.125000
    ...

where the integer index is the state (0..2^n-1) and the floating point value
is the probability (|amplitude|^2).

This script runs the executable, parses the probability lines, and draws a bar
chart using matplotlib.  The horizontal axis is the state index and the vertical
axis is the probability.

Usage::

    python3 plot_annealing.py executable_name [--output plot.png]

The executable is expected to be in the 'bin' directory.
If an output file is supplied, save the plot to that path. Otherwise, save to 'plot.png'.
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
    stdout.  Only the probability values are returned as a NumPy array ordered
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


def plot_probabilities(probs: np.ndarray, output: str = None):
    """Draw a bar chart of the probabilities.

    ``probs`` is a 1D array containing probabilities for states 0..len(probs)-1.
    If ``output`` is provided, save the figure to that path instead of displaying it.
    If not, save to 'plot.png'.
    """
    states = np.arange(probs.size)
    # Adjust figure size based on number of states
    fig_width = max(10, probs.size / 100)
    plt.figure(figsize=(fig_width, 6))
    plt.bar(states, probs, width=0.8)
    plt.xlabel("State index")
    plt.ylabel("Probability")
    plt.title("Quantum Annealing State Probabilities")
    
    # Set x-axis ticks to avoid crowding
    if probs.size > 100:
        # For large number of states, show only every Nth tick label
        tick_interval = max(1, probs.size // 20)
        tick_positions = np.arange(0, probs.size, tick_interval)
        plt.xticks(tick_positions, rotation=45, fontsize=8)
    else:
        plt.xticks(states, rotation=45)
    
    plt.tight_layout()
    if output:
        plt.savefig(output)
        print(f"Saved plot to {output}")
    else:
        output_file = "plot.png"
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Plot probabilities from a quantum annealing executable.")
    parser.add_argument("exe_name", help="Name of the compiled C executable in the 'bin' directory that prints state probabilities.")
    parser.add_argument("--output", "-o", help="If given, save the plot to this file instead of displaying it.")
    args = parser.parse_args()

    exe_path = os.path.join("bin", args.exe_name)
    if not os.path.isfile(exe_path):
        print(f"Executable not found: {exe_path}", file=sys.stderr)
        sys.exit(1)

    probs = read_probabilities_from_executable(exe_path)
    if probs.size == 0:
        print("No probabilities were found in the executable output.", file=sys.stderr)
        sys.exit(1)
    plot_probabilities(probs, args.output)


if __name__ == "__main__":
    main()
