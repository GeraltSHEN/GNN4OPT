import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_DATASET_ROOT = Path("data/data_dist/set_cover")
DEFAULT_CONFIG_NAME = "set_cover_cfg14"


def _select_field(fieldnames: Iterable[str], target: str) -> str:
    target_lower = target.lower()
    for name in fieldnames or []:
        if name.lower() == target_lower:
            return name
    raise ValueError(f"Required column `{target}` not found in CSV header.")


def _select_optional_field(fieldnames: Iterable[str], target: str) -> Optional[str]:
    try:
        return _select_field(fieldnames, target)
    except ValueError:
        return None


def _compute_average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _resolve_csv_paths(args: argparse.Namespace) -> List[Path]:
    if args.csv_path:
        return [args.csv_path]
    base = args.dataset_root
    if args.config_name:
        base = base / args.config_name
    return [base / f"{split}_samples.csv" for split in args.splits]


def _load_margin_stats(
    csv_paths: List[Path],
) -> Tuple[
    List[float],
    Dict[int, List[float]],
    List[float],
    Dict[int, List[float]],
    List[Tuple[Path, int, float, int, float]],
]:
    tied_margins: List[float] = []
    tied_by_type: Dict[int, List[float]] = {}
    unique_margins: List[float] = []
    unique_by_type: Dict[int, List[float]] = {}
    file_stats: List[Tuple[Path, int, float, int, float]] = []

    for csv_path in csv_paths:
        file_tied: List[float] = []
        file_unique: List[float] = []
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            tie_field = _select_field(reader.fieldnames, "tied_best_scores")
            margin_field = _select_field(reader.fieldnames, "first_margin")
            sample_type_field = _select_optional_field(reader.fieldnames, "sample_type")

            for row in reader:
                try:
                    ties = int(float(row[tie_field]))
                    margin = float(row[margin_field])
                except (TypeError, ValueError, KeyError):
                    continue

                if ties == 1:
                    unique_margins.append(margin)
                    file_unique.append(margin)
                    if sample_type_field:
                        try:
                            sample_type = int(float(row[sample_type_field]))
                        except (TypeError, ValueError, KeyError):
                            continue
                        unique_by_type.setdefault(sample_type, []).append(margin)
                elif ties >= 2:
                    tied_margins.append(margin)
                    file_tied.append(margin)
                    if sample_type_field:
                        try:
                            sample_type = int(float(row[sample_type_field]))
                        except (TypeError, ValueError, KeyError):
                            continue
                        tied_by_type.setdefault(sample_type, []).append(margin)

        file_stats.append(
            (
                csv_path,
                len(file_tied),
                _compute_average(file_tied),
                len(file_unique),
                _compute_average(file_unique),
            )
        )

    return tied_margins, tied_by_type, unique_margins, unique_by_type, file_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average first_margin for samples by tie count (ties >= 2 and unique best)."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing set_cover configurations (default: data/data_dist/set_cover).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=DEFAULT_CONFIG_NAME,
        help="Configuration folder inside the dataset root (default: set_cover_cfg14).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        choices=("train", "val", "test"),
        help="Splits to include when csv-path is not provided (default: train val test).",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional direct path to a *_samples.csv file. Overrides dataset-root/config-name/splits.",
    )
    args = parser.parse_args()

    csv_paths = _resolve_csv_paths(args)
    missing = [p for p in csv_paths if not p.exists()]
    existing = [p for p in csv_paths if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No CSV files found. Checked: {', '.join(str(p) for p in csv_paths)}"
        )

    if missing:
        print("Warning: Skipping missing files -> " + ", ".join(str(p) for p in missing))

    (
        tied_margins,
        tied_by_type,
        unique_margins,
        unique_by_type,
        file_stats,
    ) = _load_margin_stats(existing)
    tied_count = len(tied_margins)
    unique_count = len(unique_margins)

    print("Included files:")
    for path, tied_num, tied_avg, unique_num, unique_avg in file_stats:
        print(
            f"  {path}: tied={tied_num}, avg={tied_avg:.6f}; "
            f"unique={unique_num}, avg={unique_avg:.6f}"
        )

    print(f"Total tied samples across included files (ties >= 2): {tied_count}")
    print(f"Total unique-best samples (ties == 1): {unique_count}")

    if tied_count:
        avg_tied = _compute_average(tied_margins)
        print(f"Average first_margin (tied): {avg_tied:.6f}")
        if tied_by_type:
            print("Average first_margin by sample_type (tied):")
            for sample_type in sorted(tied_by_type):
                values = tied_by_type[sample_type]
                print(
                    f"  type {sample_type}: {len(values)} samples, avg={_compute_average(values):.6f}"
                )

    if unique_count:
        avg_unique = _compute_average(unique_margins)
        print(f"Average first_margin (unique best): {avg_unique:.6f}")
        if unique_by_type:
            print("Average first_margin by sample_type (unique best):")
            for sample_type in sorted(unique_by_type):
                values = unique_by_type[sample_type]
                print(
                    f"  type {sample_type}: {len(values)} samples, avg={_compute_average(values):.6f}"
                )


if __name__ == "__main__":
    main()
