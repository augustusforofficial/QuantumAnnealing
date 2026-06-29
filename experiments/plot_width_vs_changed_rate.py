import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("width_summary.csv")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("width_vs_changed_rate.png")

    widths = []
    rates = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            widths.append(float(row["width"]))
            rates.append(float(row["changed_rate"]))

    if not widths:
        print(f"No data found: {csv_path}")
        return 1

    plt.figure(figsize=(8, 5))
    plt.plot(widths, rates, marker="o")
    plt.xlabel("width")
    plt.ylabel("changed rate")
    plt.title("Width vs changed rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved graph to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
