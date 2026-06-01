#!/usr/bin/env python3
"""
Create a quick monthly-generation plot for the example PV plants.

This is a lightweight visualization helper for first-time users who want to
inspect the included sample data before running the benchmark.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "figures"
PLANT_IDS = ("171", "172", "186")


def load_monthly_generation(plant_id: str, year: int) -> pd.Series:
    """Return monthly generated electricity for one plant and year."""
    path = DATA_DIR / f"Project{plant_id}.csv"
    df = pd.read_csv(path, usecols=["Year", "Month", "Electricity Generated"])
    df = df[df["Year"].astype(int) == year].copy()
    df["Month"] = df["Month"].astype(int)
    monthly = df.groupby("Month")["Electricity Generated"].sum()
    return monthly.reindex(range(1, 13), fill_value=0.0)


def main() -> None:
    year = 2023
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for plant_id in PLANT_IDS:
        monthly = load_monthly_generation(plant_id, year)
        ax.plot(monthly.index, monthly.values, marker="o", linewidth=2, label=f"Plant {plant_id}")

    ax.set_xlabel("Month")
    ax.set_ylabel("Electricity Generated (kWh)")
    ax.set_title(f"Monthly PV Generation For Included Example Plants ({year})")
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_path = OUTPUT_DIR / "example_monthly_generation.png"
    fig.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
