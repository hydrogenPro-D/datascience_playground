"""Summary of strike_times.csv (produced by strike_analysis.py):
electrode count, electrodes outside limits, and average step times."""
from pathlib import Path
import pandas as pd

CSV_FILE = Path(__file__).resolve().parent / "strike_times.csv"


def main():
    df = pd.read_csv(CSV_FILE)

    n_electrodes = len(df)

    # Outside limits: actual exceeded set time in either strike pass (strict, any second over)
    over1 = df["Actual_time_strike1"] > df["Set_time_strike1"]
    over2 = df["Actual_time_strike2"] > df["Set_time_strike2"]
    outside = over1 | over2

    print("=" * 50)
    print("Strike summary")
    print("=" * 50)
    print(f"Electrodes:                     {n_electrodes}")
    print(f"Outside limits (either pass):   {int(outside.sum())} "
          f"({outside.mean():.1%})")
    print(f"  - strike pass 1 over set:     {int(over1.sum())}")
    print(f"  - strike pass 2 over set:     {int(over2.sum())}")

    print("\nAverage times (seconds):")
    print(f"  Strike pass 1: set {df['Set_time_strike1'].mean():.2f}  "
          f"actual {df['Actual_time_strike1'].mean():.2f}")
    print(f"  Strike pass 2: set {df['Set_time_strike2'].mean():.2f}  "
          f"actual {df['Actual_time_strike2'].mean():.2f}")
    print(f"  Both passes total: set {df['Set_time_total'].mean():.2f}  "
          f"actual {df['Actual_time_total'].mean():.2f}")

    over1_amount = (df["Actual_time_strike1"] - df["Set_time_strike1"])[over1]
    over2_amount = (df["Actual_time_strike2"] - df["Set_time_strike2"])[over2]
    print("\nAverage overrun when over (seconds):")
    print(f"  Strike pass 1: {over1_amount.mean():.2f}  (n={len(over1_amount)})")
    print(f"  Strike pass 2: {over2_amount.mean():.2f}  (n={len(over2_amount)})")


if __name__ == "__main__":
    main()
