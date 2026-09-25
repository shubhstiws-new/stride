#!/usr/bin/env python3
"""Render the scale-sweep results as a PNG for the README.

Usage:
    python benchmarks/plot_scale_sweep.py \
        --results benchmarks/results/scale_sweep_canonical_20260220.json \
        --out docs/scale_sweep.png
"""

import argparse
import json

import matplotlib.pyplot as plt

# Fixed order: each engine keeps its colour and marker regardless of which runs succeed.
ENGINES = [
    ("PySpark", "#2a78d6", "o"),
    ("Polars", "#eb6834", "s"),
    ("DuckDB", "#1baf7a", "D"),
    ("Pandas", "#eda100", "^"),
    ("Dask", "#e87ba4", "v"),
]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sweep = json.load(open(args.results))["sweep"]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    failures = []
    for name, color, marker in ENGINES:
        ok = [(s["approx_data_size_gb"], s["frameworks"][name]["total_time_s"])
              for s in sweep if s["frameworks"][name]["status"] == "OK"]
        failed = [s for s in sweep if s["frameworks"][name]["status"] != "OK"]
        xs, ys = zip(*ok)
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        if failed:
            first = failed[0]
            failures.append(f"{name} at {first['approx_data_size_gb']:g} GB "
                            f"({first['frameworks'][name]['status']})")

    ax.annotate("PySpark: only engine to complete\n30 GB (2,354 s)", (29.8, 2353.8),
                xytext=(-120, 14), textcoords="offset points", fontsize=8.5, color=INK)

    ax.text(0.99, 0.03, "Did not complete:\n" + "\n".join(failures), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=MUTED, linespacing=1.4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input size (GB, log scale)", color=MUTED, fontsize=9)
    ax.set_ylabel("Wall-clock time (s, log scale)", color=MUTED, fontsize=9)
    ax.set_title("Session detection wall-clock time vs. input size (single node)",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", ncol=5)
    ax.set_ylim(0.1, 40000)

    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
