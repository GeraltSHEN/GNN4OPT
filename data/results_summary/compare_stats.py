#!/usr/bin/env python3
"""
Utilities to compare experiment statistics and configuration parameters across
dataset-specific runs.

Usage:
    python compare_stats.py --dataset set_cover
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ConfigInfo:
    """Metadata for a dataset configuration."""

    index: int
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise stats and config parameters for dataset runs."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. set_cover). Matches folders like <dataset>_cfg1.",
    )
    return parser.parse_args()


def discover_config_dirs(base_dir: Path, dataset: str) -> List[ConfigInfo]:
    """Return ordered config directories for the dataset."""
    prefix = f"{dataset.lower()}_cfg"
    configs: List[ConfigInfo] = []
    for candidate in base_dir.iterdir():
        if not candidate.is_dir():
            continue
        name_lower = candidate.name.lower()
        if not name_lower.startswith(prefix):
            continue
        suffix = candidate.name[len(prefix):]
        match = re.search(r"(\d+)", suffix)
        if not match:
            continue
        index = int(match.group(1))
        # if index == 0:
        #     continue
        configs.append(ConfigInfo(index=index, name=candidate.name, path=candidate))
    configs.sort(key=lambda cfg: cfg.index)
    return configs


def collect_split_metrics(split_dir: Path) -> Dict[str, object]:
    """Load metrics from JSON files in a split directory, prefixing columns."""
    metrics: Dict[str, object] = {}
    for json_file in sorted(split_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError:
            continue
        prefix = f"{split_dir.name}_{json_file.stem}"
        if isinstance(data, dict):
            for key, value in data.items():
                metrics[f"{prefix}_{key}"] = value
        else:
            metrics[prefix] = data
    return metrics


def gather_stats(configs: Iterable[ConfigInfo]) -> Tuple[List[Dict[str, object]], List[str]]:
    """Collect stats rows and determine the union of metric columns."""
    rows: List[Dict[str, object]] = []
    metric_columns: List[str] = []

    for cfg in configs:
        row: Dict[str, object] = {"cfg": cfg.name, "cfg_index": cfg.index}
        for split_dir in sorted(p for p in cfg.path.iterdir() if p.is_dir()):
            split_metrics = collect_split_metrics(split_dir)
            for column, value in split_metrics.items():
                if column not in metric_columns:
                    metric_columns.append(column)
                row[column] = value
        rows.append(row)

    return rows, metric_columns


def write_stats_csv(output_path: Path, rows: List[Dict[str, object]], metric_columns: List[str]) -> None:
    """Write the stats summary CSV."""
    fieldnames = ["cfg_index", "cfg"] + metric_columns
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {
                key: _format_metric_value(key, row.get(key, ""))
                for key in fieldnames
            }
            writer.writerow(formatted)


def locate_param_file(cfg_root: Path, dataset: str, cfg_index: int, cfg_name: str) -> Optional[Path]:
    """Attempt to find a parameter file associated with the cfg."""
    candidates = [
        cfg_root / f"{dataset}_{cfg_index}",
        cfg_root / f"{dataset}_{cfg_index}.yaml",
        cfg_root / f"{dataset}_{cfg_index}.yml",
        cfg_root / dataset / f"{cfg_index}",
        cfg_root / dataset / f"{cfg_index}.yaml",
        cfg_root / dataset / f"{cfg_index}.yml",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            for suffix in ("config.yaml", "config.yml", "cfg.yaml", "cfg.yml", "params.yaml"):
                maybe = candidate / suffix
                if maybe.is_file():
                    return maybe

    pattern = re.compile(
        rf"^{re.escape(dataset)}.*{cfg_index}(?:[_-][A-Za-z0-9]+|[A-Za-z][A-Za-z0-9_-]*)*(?:\.[A-Za-z0-9]+)?$",
        re.IGNORECASE,
    )
    matches = [
        path for path in cfg_root.iterdir()
        if (path.is_file() or path.is_dir()) and pattern.match(path.name)
    ]
    if len(matches) == 1 and matches[0].is_file():
        return matches[0]
    if len(matches) == 1 and matches[0].is_dir():
        for suffix in ("config.yaml", "config.yml", "cfg.yaml", "cfg.yml", "params.yaml"):
            maybe = matches[0] / suffix
            if maybe.is_file():
                return maybe
    return None


def parse_parameter_file(path: Path) -> Dict[str, object]:
    """Parse a simple key: value parameter file."""
    params: Dict[str, object] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("'") and value.endswith("'") and len(value) >= 2:
            value = value[1:-1]
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        if value.lower() in ("true", "false"):
            params[key] = value.lower() == "true"
            continue
        if value.lower() in ("none", "null"):
            params[key] = ""
            continue
        if value == "":
            params[key] = ""
            continue
        try:
            params[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            params[key] = float(value)
            continue
        except ValueError:
            pass
        params[key] = value
    return params


def gather_parameters(repo_root: Path, dataset: str, configs: Iterable[ConfigInfo]) -> Tuple[Dict[str, Dict[str, object]], List[str]]:
    """Load parameter dictionaries keyed by config name."""
    cfg_root = repo_root / "cfg"
    parameters: Dict[str, Dict[str, object]] = {}
    missing_params: List[str] = []

    for cfg in configs:
        param_file = locate_param_file(cfg_root, dataset, cfg.index, cfg.name)
        if param_file is None:
            missing_params.append(cfg.name)
            parameters[cfg.name] = {}
            continue
        if param_file.is_dir():
            parameters[cfg.name] = {}
            missing_params.append(cfg.name)
            continue
        parameters[cfg.name] = parse_parameter_file(param_file)

    if missing_params:
        print(
            f"Warning: No parameter file found for {', '.join(missing_params)}. Columns will be empty."
        )

    return parameters, sorted(parameters.keys(), key=lambda name: name)


def write_parameters_csv(
    output_path: Path,
    param_values: Dict[str, Dict[str, object]],
    cfg_names: List[str],
) -> Tuple[List[str], List[Dict[str, object]]]:
    """Write CSV summarising parameter differences across configs."""
    all_keys: List[str] = []
    seen = set()
    for params in param_values.values():
        for key in params:
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    fieldnames = ["parameter"] + cfg_names + ["variation"]
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        rows: List[Dict[str, object]] = []
        for key in all_keys:
            row = {"parameter": key}
            values = []
            for cfg_name in cfg_names:
                value = param_values[cfg_name].get(key, "")
                row[cfg_name] = value
                values.append(value)
            unique_values = {json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v) for v in values}
            row["variation"] = "persistent" if len(unique_values) == 1 else "different"
            rows.append(row)

        rows.sort(key=lambda item: (0 if item["variation"] == "different" else 1, item["parameter"]))
        writer.writerows(rows)
    return fieldnames, rows


def _format_metric_value(column: str, value: object) -> str:
    """Pretty print metric values for terminal output."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)

    lower_col = column.lower()
    if "n_samples" in lower_col:
        return str(int(round(numeric)))
    if any(token in lower_col for token in ("loss", "score_diff", "normalized_score_diff")):
        return f"{numeric:.2f}"
    if "accuracy" in lower_col:
        return f"{numeric * 100:.2f}%"
    return str(value)


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a simple table with padded columns."""
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    lines = [format_row(headers)]
    for row in rows:
        lines.append(format_row(row))
    return "\n".join(lines)


def print_stats_table(columns: Sequence[Tuple[str, str]], rows: List[Dict[str, object]]) -> None:
    """Print the stats table using friendly formatting."""
    printable_rows: List[List[str]] = []
    for row in rows:
        printable_rows.append(
            [
                _format_metric_value(column, row.get(column, ""))
                for column, _ in columns
            ]
        )
    headers = [label for _, label in columns]
    print("\n=== Stats comparison ===")
    print(_render_table(headers, printable_rows))


def _format_parameter_value(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def print_parameter_table(fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    """Print the parameter comparison table."""
    if not rows:
        print("\n=== Parameter comparison ===")
        print("(no differing parameters)")
        return
    printable_rows: List[List[str]] = []
    for row in rows:
        printable_rows.append([_format_parameter_value(row.get(field, "")) for field in fieldnames])
    print("\n=== Parameter comparison ===")
    print(_render_table(fieldnames, printable_rows))


def main() -> None:
    args = parse_args()
    dataset = args.dataset

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent.parent

    configs = discover_config_dirs(base_dir, dataset)
    if not configs:
        raise SystemExit(f"No config folders found for dataset '{dataset}'.")

    stats_rows, metric_columns = gather_stats(configs)
    stats_output = base_dir / f"{dataset}_stats_comparison.csv"
    write_stats_csv(stats_output, stats_rows, metric_columns)
    stats_display_columns = [
        ("cfg_index", "cfg"),
        ("test_eval_acc_Accuracy", "accuracy"),
        ("test_eval_acc_Top5_Accuracy", "top5 accuracy"),
    ]
    print_stats_table(stats_display_columns, stats_rows)

    param_values, cfg_names = gather_parameters(repo_root, dataset, configs)
    params_output = base_dir / f"{dataset}_cfg_parameters.csv"
    param_fieldnames, param_rows = write_parameters_csv(params_output, param_values, cfg_names)
    differing_rows = [row for row in param_rows if row.get("variation") == "different"]
    print_parameter_table(param_fieldnames, differing_rows)

    rel_stats = stats_output.relative_to(repo_root)
    rel_params = params_output.relative_to(repo_root)
    print(f"Stats comparison written to {rel_stats}")
    print(f"Parameter comparison written to {rel_params}")


if __name__ == "__main__":
    main()
