import argparse
import time
from pathlib import Path

import numpy as np

from heuristics.utils import (
    SCIPBranchingContext,
    UNIVERSAL_CUTOFFBOUND_DEFAULT,
    compute_strong_branch_score,
    load_sample,
    resolve_effective_cutoffbound,
    unpack_sample_data,
)


DEFAULT_DATA_DIR = Path(
    "/scratch/gilbreth/chen4433/GNN4OPT/legacy_code_generator/data/samples/"
    "setcover_xCutoffBound/500r_1000c_0.05d/test"
)


def objective_or_nan(result) -> float:
    if result.success and np.isfinite(float(result.objective_value)):
        return float(result.objective_value)
    return float("nan")


def fmt_value(x: float) -> str:
    return "nan" if not np.isfinite(float(x)) else f"{float(x):.8g}"


def main():
    parser = argparse.ArgumentParser(description="Quick one-off check for dual option consistency (1 vs 3, 2 vs 4).")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--max_candidates", type=int, default=2)
    parser.add_argument("--time_limit_sec", type=float, default=120.0)
    parser.add_argument("--universal_cutoffbound", type=float, default=UNIVERSAL_CUTOFFBOUND_DEFAULT)
    args = parser.parse_args()

    sample_files = sorted(args.data_dir.glob("sample_*.pkl"))[: int(args.max_samples)]
    if not sample_files:
        raise RuntimeError(f"No sample_*.pkl found under: {args.data_dir}")

    for sample_path in sample_files:
        raw_sample = load_sample(sample_path)
        sample_record = unpack_sample_data(raw_sample["data"])
        context = SCIPBranchingContext.from_sample_state(sample_record.sample_state)
        parent = context.solve_parent_primal()
        if (not parent.success) or (parent.objective_value is None):
            print(f"\n=== {sample_path.name} ===")
            print(f"parent LP failed: status={parent.status} message={parent.message}")
            continue

        parent_obj = float(parent.objective_value)
        cutoffbound = float(sample_record.cutoffbound)
        candidates = np.asarray(sample_record.action_set, dtype=np.int64)[: max(1, int(args.max_candidates))]

        print(f"\n=== {sample_path.name} ===")
        print(
            f"parent_obj={parent_obj:.8g} sample_cutoffbound={cutoffbound:.8g} "
            f"candidates={candidates.tolist()}"
        )

        for var_idx in candidates.tolist():
            option_results = {}
            option_times = {}
            for option in (1, 3, 2, 4):
                t0 = time.perf_counter()
                down = context.solve_child_dual(
                    int(var_idx),
                    direction="down",
                    cutoffbound=cutoffbound,
                    dual_option=int(option),
                    universal_cutoffbound=float(args.universal_cutoffbound),
                    time_limit_sec=float(args.time_limit_sec),
                )
                up = context.solve_child_dual(
                    int(var_idx),
                    direction="up",
                    cutoffbound=cutoffbound,
                    dual_option=int(option),
                    universal_cutoffbound=float(args.universal_cutoffbound),
                    time_limit_sec=float(args.time_limit_sec),
                )
                option_times[option] = time.perf_counter() - t0
                option_results[option] = (down, up)

            option_scores = {}
            for option in (1, 3, 2, 4):
                eff_cutoff = float(
                    resolve_effective_cutoffbound(
                        cutoffbound=cutoffbound,
                        dual_option=int(option),
                        universal_cutoffbound=float(args.universal_cutoffbound),
                    )
                )
                down, up = option_results[option]
                down_obj = objective_or_nan(down)
                up_obj = objective_or_nan(up)
                if np.isfinite(down_obj) and np.isfinite(up_obj):
                    option_scores[option] = float(
                        compute_strong_branch_score(
                            parent_obj=parent_obj,
                            child_one_obj=up_obj,
                            child_zero_obj=down_obj,
                            cutoffbound=eff_cutoff,
                        )
                    )
                else:
                    option_scores[option] = float("nan")

            print(f"var={int(var_idx)}")
            for option in (1, 3, 2, 4):
                down_obj = objective_or_nan(option_results[option][0])
                up_obj = objective_or_nan(option_results[option][1])
                print(
                    f"  opt{option}: t={option_times[option]:.3f}s "
                    f"down={fmt_value(down_obj)} up={fmt_value(up_obj)} "
                    f"score={fmt_value(option_scores[option])}"
                )

            d13_down = abs(objective_or_nan(option_results[1][0]) - objective_or_nan(option_results[3][0]))
            d13_up = abs(objective_or_nan(option_results[1][1]) - objective_or_nan(option_results[3][1]))
            d13_score = abs(option_scores[1] - option_scores[3])
            d24_down = abs(objective_or_nan(option_results[2][0]) - objective_or_nan(option_results[4][0]))
            d24_up = abs(objective_or_nan(option_results[2][1]) - objective_or_nan(option_results[4][1]))
            d24_score = abs(option_scores[2] - option_scores[4])
            print(
                f"  diff(1,3): down={fmt_value(d13_down)} up={fmt_value(d13_up)} "
                f"score={fmt_value(d13_score)}"
            )
            print(
                f"  diff(2,4): down={fmt_value(d24_down)} up={fmt_value(d24_up)} "
                f"score={fmt_value(d24_score)}"
            )


if __name__ == "__main__":
    main()
