import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from default_args import DATASET_DEFAULTS
from eval import _load_config, _merge_args_with_config

CSV_FILES = ("train_samples.csv", "val_samples.csv", "test_samples.csv")
MARGIN_THRESHOLDS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
MARGIN_LABELS = (
    "0~1e-1",
    "1e-1~1e-2",
    "1e-2~1e-3",
    "1e-3~1e-4",
    "1e-4~1e-5",
    "1e-5~",
)

HOLO_VALUES = (1, 2, 3)
HOLO_GROUP_KEYS = tuple(f"sel{sel}_holo{holo}" for sel in HOLO_VALUES for holo in HOLO_VALUES)
HOLO_GROUP_LABELS = {
    key: f"selector={sel} & holo={holo}"
    for sel in HOLO_VALUES
    for holo in HOLO_VALUES
    for key in (f"sel{sel}_holo{holo}",)
}


def _resolve_model_dir(args: argparse.Namespace) -> Path:
    base_dir = Path(getattr(args, "model_dir", "./models"))
    model_name = getattr(args, "model", None)
    if model_name:
        base_dir = base_dir / model_name
    model_id = getattr(args, "model_id", None)
    if model_id:
        base_dir = base_dir / model_id
    model_suffix = getattr(args, "model_suffix", "")
    if model_suffix:
        base_dir = Path(f"{base_dir}_{model_suffix}")
    return base_dir


def _build_run_root(
    dataset: str, cfg_idx: int, config_root: Path, parent_dir: Path, model_suffix: str
) -> Path:
    cfg = _load_config(config_root, dataset, cfg_idx)
    init_args = argparse.Namespace(
        dataset=dataset,
        cfg_idx=cfg_idx,
        config_root=str(config_root),
        model_suffix=model_suffix,
    )
    args = _merge_args_with_config(init_args, cfg)
    run_name = _resolve_model_dir(args).name
    return parent_dir / dataset / run_name


def _select_tie_field(fieldnames) -> str:
    for name in fieldnames or []:
        lower = name.lower()
        if lower.startswith("tied_best"):
            return name
    raise ValueError("No tie-count column found in CSV.")


def _select_field(fieldnames: Iterable[str], target: str) -> str:
    target_lower = target.lower()
    for name in fieldnames or []:
        if name.lower() == target_lower:
            return name
    raise ValueError(f"No `{target}` column found in CSV.")


def _select_sample_type_field(fieldnames: Iterable[str]) -> str:
    lower_map = {name.lower(): name for name in fieldnames or []}
    for candidate in ("sample_type_selector", "sample_type"):
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError("No sample type column found in CSV.")


def _select_optional_field(fieldnames: Iterable[str], target: str) -> Optional[str]:
    try:
        return _select_field(fieldnames, target)
    except ValueError:
        return None


def _bin_margins(values: Iterable[float]) -> Tuple[str, ...]:
    counts = [0] * len(MARGIN_LABELS)
    for margin in values:
        if margin >= MARGIN_THRESHOLDS[0]:
            counts[0] += 1
        elif margin >= MARGIN_THRESHOLDS[1]:
            counts[1] += 1
        elif margin >= MARGIN_THRESHOLDS[2]:
            counts[2] += 1
        elif margin >= MARGIN_THRESHOLDS[3]:
            counts[3] += 1
        elif margin >= MARGIN_THRESHOLDS[4]:
            counts[4] += 1
        else:
            counts[5] += 1
    return tuple(counts)


def collect_distributions(
    root: Path,
) -> Tuple[
    Dict[int, Counter],
    Dict[int, list],
    Dict[str, Dict[str, int]],
    Dict[str, Counter],
    Dict[str, list],
    bool,
]:
    tie_counts: Dict[int, Counter] = {1: Counter(), 2: Counter(), 3: Counter()}
    margins: Dict[int, list] = {1: [], 2: [], 3: []}
    holo_tie_counts: Dict[str, Counter] = {key: Counter() for key in HOLO_GROUP_KEYS}
    holo_margins: Dict[str, list] = {key: [] for key in HOLO_GROUP_KEYS}
    summaries: Dict[str, Dict[str, int]] = {}
    has_holo_data = False

    for name in CSV_FILES:
        csv_path = root / name
        if not csv_path.exists():
            summaries[name] = None
            continue

        total = le_8 = gt_8 = 0
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            tie_field = _select_tie_field(reader.fieldnames)
            margin_field = _select_field(reader.fieldnames, "first_margin")
            sample_type_field = _select_sample_type_field(reader.fieldnames)
            selector_field = _select_optional_field(reader.fieldnames, "sample_type_selector")
            holo_field = _select_optional_field(reader.fieldnames, "sample_type_holo")

            for row in reader:
                try:
                    ties = int(float(row[tie_field]))
                    margin = float(row[margin_field])
                    sample_type = int(float(row[sample_type_field]))
                    selector_type = (
                        int(float(row[selector_field])) if selector_field else None
                    )
                    holo_type = int(float(row[holo_field])) if holo_field else None
                except (TypeError, ValueError, KeyError):
                    continue

                total += 1
                if ties <= 8:
                    le_8 += 1
                else:
                    gt_8 += 1

                if sample_type in tie_counts:
                    tie_counts[sample_type][ties] += 1
                    margins[sample_type].append(margin)

                if selector_type is not None and holo_type is not None:
                    holo_key = f"sel{selector_type}_holo{holo_type}"
                    if holo_key in holo_tie_counts:
                        has_holo_data = True
                        holo_tie_counts[holo_key][ties] += 1
                        holo_margins[holo_key].append(margin)

        summaries[name] = {"total": total, "le_8": le_8, "gt_8": gt_8}

    return tie_counts, margins, summaries, holo_tie_counts, holo_margins, has_holo_data


def _build_tie_accum_table(
    tie_counts: Dict,
    key_order: Optional[Sequence] = None,
    label_map: Optional[Dict] = None,
) -> Tuple[list, list, Dict[str, int], int]:
    headers = [
        "sample_type",
        "# no tied",
        "# tied = 2",
        "# tied <= 3",
        "# tied <= 4",
        "# tied <= 5",
        "# tied <= 6",
        "# tied <= 7",
        "# tied <= 8",
        "# tied <= 9 or more",
        "# no tied & # tied",
    ]

    rows: list = []
    sample_totals: Dict[str, int] = {}
    keys = key_order or sorted(tie_counts.keys())
    for sample_type in keys:
        counter = tie_counts.get(sample_type, Counter())
        counts = [
            counter.get(1, 0),
            counter.get(2, 0),
        ]
        for upper in range(3, 9):
            counts.append(sum(v for tie, v in counter.items() if 2 <= tie <= upper))
        counts.append(sum(v for tie, v in counter.items() if tie >= 2))
        display_key = label_map.get(sample_type, sample_type) if label_map else sample_type
        row_total = sum(counter.values())
        row = [str(display_key)] + counts + [row_total]
        rows.append(row)
        sample_totals[str(display_key)] = row_total
    total_samples = sum(sample_totals.values())
    return headers, rows, sample_totals, total_samples


def _format_table(headers: list, rows: list) -> list:
    str_rows = [[str(cell) for cell in row] for row in rows]
    columns = [headers] + str_rows
    widths = [max(len(col[i]) for col in columns) for i in range(len(headers))]

    lines = []
    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    separator = "-+-".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header_line)
    lines.append(separator)
    for row in str_rows:
        line = " | ".join(
            row[i].ljust(widths[i]) if i == 0 else row[i].rjust(widths[i])
            for i in range(len(headers))
        )
        lines.append(line)
    return lines


def _percent_values(values: list, denom: int) -> list:
    if denom <= 0:
        return ["0.00" for _ in values]
    return [f"{(val / denom) * 100:.2f}" for val in values]


def _save_table_csv(
    headers: list,
    rows: list,
    sample_totals: Dict[str, int],
    total_samples: int,
    output_path: Path,
) -> None:
    row_strings = [[str(cell) for cell in row] for row in rows]
    rowwise_percent = [
        [str(row[0])] + _percent_values(row[1:], sample_totals.get(row[0], 0))
        for row in rows
    ]
    total_percent = [
        [str(row[0])] + _percent_values(row[1:], total_samples) for row in rows
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Counts"])
        writer.writerow(headers)
        writer.writerows(row_strings)
        writer.writerow([])
        writer.writerow(["Row-wise percentage (of sample_type total, %)"])
        writer.writerow(headers)
        writer.writerows(rowwise_percent)
        writer.writerow([])
        writer.writerow(["Total percentage (overall % of all samples)"])
        writer.writerow(headers)
        writer.writerows(total_percent)


def plot_distributions(
    tie_counts: Dict,
    margins: Dict,
    output_path: Path,
    tie_axis_override: Optional[Sequence[int]] = None,
    show: bool = False,
) -> None:
    if not any(tie_counts.values()):
        print("No data available to plot.")
        return

    if tie_axis_override:
        tie_axis = list(tie_axis_override)
    else:
        max_ties = max(
            (max(counter) if counter else 0 for counter in tie_counts.values())
        )
        tie_axis = list(range(max_ties + 1))

    n_rows = len(tie_counts)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for idx, sample_type in enumerate(tie_counts.keys()):
        ax_tie = axes[idx][0]
        tie_values = [tie_counts[sample_type].get(x, 0) for x in tie_axis]
        ax_tie.bar(tie_axis, tie_values, color="#4C72B0")
        ax_tie.set_ylabel(f"sample_type {sample_type}")
        if tie_axis:
            ax_tie.set_xlim(tie_axis[0] - 0.5, tie_axis[-1] + 0.5)
        if idx == len(tie_counts) - 1:
            ax_tie.set_xlabel("number of tied best scores")
        if idx == 0:
            ax_tie.set_title("Tied best scores distribution")

        ax_margin = axes[idx][1]
        margin_counts = _bin_margins(margins[sample_type])
        positions = range(len(MARGIN_LABELS))
        ax_margin.bar(positions, margin_counts, color="#55A868")
        ax_margin.set_xticks(list(positions))
        ax_margin.set_xticklabels(MARGIN_LABELS, rotation=25, ha="right")
        if idx == len(tie_counts) - 1:
            ax_margin.set_xlabel("first margin segments")
        if idx == 0:
            ax_margin.set_title("First margin distribution")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute tie distribution stats for sample CSV files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="set_cover",
        choices=sorted(DATASET_DEFAULTS.keys()),
        help="Dataset to load configurations from.",
    )
    parser.add_argument(
        "--cfg_idx",
        type=int,
        nargs="+",
        help="Configuration indices to analyze (one directory per cfg).",
    )
    parser.add_argument(
        "--config_root",
        type=Path,
        default=Path("./cfg"),
        help="Root directory containing configuration files.",
    )
    parser.add_argument(
        "--parent_dir",
        type=Path,
        default=Path("data/data_dist"),
        help="Parent directory where verify_dataset outputs are stored.",
    )
    parser.add_argument(
        "--model_suffix",
        type=str,
        default="",
        help="Optional model suffix used when generating the stats.",
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="*",
        help="Explicit directories containing *_samples.csv files. Overrides cfg_idx if provided.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    args = parser.parse_args()

    if args.roots:
        run_roots = list(args.roots)
    else:
        if not args.cfg_idx:
            raise ValueError("Provide at least one --cfg_idx or explicit --roots.")
        run_roots = [
            _build_run_root(
                args.dataset,
                cfg_idx,
                args.config_root,
                args.parent_dir,
                args.model_suffix,
            )
            for cfg_idx in args.cfg_idx
        ]

    for run_root in run_roots:
        print(f"Processing: {run_root}")
        tie_counts, margins, summaries, holo_tie_counts, holo_margins, has_holo = collect_distributions(
            run_root
        )
        for name in CSV_FILES:
            stats = summaries.get(name)
            if stats is None:
                print(f"{name}: missing ({run_root / name})")
            else:
                print(
                    f"{name}: total={stats['total']}, <=8 ties={stats['le_8']}, >8 ties={stats['gt_8']}"
                )

        headers, rows, sample_totals, total_samples = _build_tie_accum_table(tie_counts)
        table_output_path = run_root / "data_distribution_table.csv"
        _save_table_csv(headers, rows, sample_totals, total_samples, table_output_path)
        output_path = run_root / "data_distribution.png"
        plot_distributions(tie_counts, margins, output_path, show=args.show)
        zoom_output_path = run_root / "data_distribution_zoomed.png"
        plot_distributions(
            tie_counts,
            margins,
            zoom_output_path,
            tie_axis_override=range(1, 11),
            show=args.show,
        )

        if has_holo:
            holo_headers, holo_rows, holo_sample_totals, holo_total_samples = _build_tie_accum_table(
                holo_tie_counts, key_order=HOLO_GROUP_KEYS, label_map=HOLO_GROUP_LABELS
            )
            holo_table_path = run_root / "holo_data_distribution_table.csv"
            _save_table_csv(
                holo_headers, holo_rows, holo_sample_totals, holo_total_samples, holo_table_path
            )
            holo_output_path = run_root / "holo_data_distribution.png"
            plot_distributions(holo_tie_counts, holo_margins, holo_output_path, show=args.show)
            holo_zoom_output_path = run_root / "holo_data_distribution_zoomed.png"
            plot_distributions(
                holo_tie_counts,
                holo_margins,
                holo_zoom_output_path,
                tie_axis_override=range(1, 11),
                show=args.show,
            )


if __name__ == "__main__":
    main()
