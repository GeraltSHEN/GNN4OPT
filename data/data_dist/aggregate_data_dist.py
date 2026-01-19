#!/usr/bin/env python3
"""Aggregate distribution outputs across datasets and configurations."""

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def _read_sectioned_table(path: Path) -> pd.DataFrame:
    """Parse data_distribution_table.csv with its multi-section layout."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    records: List[dict] = []
    idx = 0
    while idx < len(rows):
        row = rows[idx]
        if not any(cell.strip() for cell in row):
            idx += 1
            continue
        if len(row) == 1:
            section = row[0].strip().strip('"')
            idx += 1
        else:
            raise ValueError(f"Unexpected row while looking for section label: {row}")

        if idx >= len(rows):
            break
        headers = rows[idx]
        idx += 1

        while idx < len(rows) and len(rows[idx]) > 1:
            data_row = rows[idx]
            record = {"section": section}
            for col_idx, header in enumerate(headers):
                record[header] = data_row[col_idx] if col_idx < len(data_row) else ""
            records.append(record)
            idx += 1

        while idx < len(rows) and not any(cell.strip() for cell in rows[idx]):
            idx += 1

    return pd.DataFrame.from_records(records)


def _iter_dataset_configs(root: Path) -> Iterable[Tuple[str, Path]]:
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        for cfg_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            yield dataset, cfg_dir


def collect_frames(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[str, str, str]]]:
    """Collect data_distribution_table and no_tie_expected_top8_summary frames."""
    dist_frames: List[pd.DataFrame] = []
    no_tie_frames: List[pd.DataFrame] = []
    missing: List[Tuple[str, str, str]] = []

    for dataset, cfg_dir in _iter_dataset_configs(root):
        cfg_name = cfg_dir.name
        dist_path = cfg_dir / "data_distribution_table.csv"
        no_tie_path = cfg_dir / "no_tie_expected_top8_summary.csv"

        if dist_path.exists():
            dist_df = _read_sectioned_table(dist_path)
            if not dist_df.empty:
                dist_df.insert(0, "cfg", cfg_name)
                dist_df.insert(0, "dataset", dataset)
                dist_frames.append(dist_df)
        else:
            missing.append((dataset, cfg_name, dist_path.name))

        if no_tie_path.exists():
            no_tie_df = pd.read_csv(no_tie_path)
            if not no_tie_df.empty:
                no_tie_df.insert(0, "cfg", cfg_name)
                no_tie_df.insert(0, "dataset", dataset)
                no_tie_frames.append(no_tie_df)
        else:
            missing.append((dataset, cfg_name, no_tie_path.name))

    dist_out = pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame()
    no_tie_out = pd.concat(no_tie_frames, ignore_index=True) if no_tie_frames else pd.DataFrame()
    return dist_out, no_tie_out, missing


def write_outputs(dist_df: pd.DataFrame, no_tie_df: pd.DataFrame, root: Path, prefix: str) -> None:
    dist_csv = root / f"{prefix}_data_distribution_table.csv"
    no_tie_csv = root / f"{prefix}_no_tie_expected_top8_summary.csv"
    excel_path = root / f"{prefix}.xlsx"

    if not dist_df.empty:
        dist_df.to_csv(dist_csv, index=False)
    if not no_tie_df.empty:
        no_tie_df.to_csv(no_tie_csv, index=False)

    if dist_df.empty and no_tie_df.empty:
        return

    datasets = sorted(
        set(dist_df["dataset"].unique().tolist() if not dist_df.empty else [])
        | set(no_tie_df["dataset"].unique().tolist() if not no_tie_df.empty else [])
    )

    with pd.ExcelWriter(excel_path) as writer:
        for dataset in datasets:
            row_cursor = 0
            if not dist_df.empty:
                subset = dist_df[dist_df["dataset"] == dataset].drop(columns=["dataset"])
                sort_cols = [col for col in ("cfg", "section", "sample_type") if col in subset.columns]
                if sort_cols:
                    subset = subset.sort_values(sort_cols)
                subset.to_excel(writer, sheet_name=dataset, index=False, startrow=row_cursor)
                row_cursor += len(subset) + 2
            if not no_tie_df.empty:
                subset = no_tie_df[no_tie_df["dataset"] == dataset].drop(columns=["dataset"])
                if row_cursor > 0:
                    row_cursor += 1
                sort_cols = [col for col in ("cfg", "sample_type") if col in subset.columns]
                if sort_cols:
                    subset = subset.sort_values(sort_cols)
                subset.to_excel(writer, sheet_name=dataset, index=False, startrow=row_cursor)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine data_distribution_table and no_tie summaries across cfgs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root folder containing dataset subdirectories (default: folder of this script).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="combined_data_dist",
        help="Prefix for the generated CSV/XLSX files.",
    )
    args = parser.parse_args()

    dist_df, no_tie_df, missing = collect_frames(args.root)
    write_outputs(dist_df, no_tie_df, args.root, args.output_prefix)

    if missing:
        print("Missing files:")
        for dataset, cfg, name in missing:
            print(f"  {dataset}/{cfg}/{name}")


if __name__ == "__main__":
    main()
