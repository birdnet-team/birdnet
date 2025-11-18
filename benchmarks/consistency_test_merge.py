from pathlib import Path

import pandas as pd

REPORT_DIR = Path("benchmarks/consistency_reports")


def merge_results() -> pd.DataFrame:
  df_report = pd.DataFrame()
  files = (
    p.absolute()
    for p in REPORT_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() == ".csv" and p.parent != REPORT_DIR
  )
  for file in files:
    df_part = pd.read_csv(file)
    df_report = pd.concat([df_report, df_part], ignore_index=True)
  df_report.to_csv(REPORT_DIR / "report.csv", index=False)
  return df_report


if __name__ == "__main__":
  merge_results()
