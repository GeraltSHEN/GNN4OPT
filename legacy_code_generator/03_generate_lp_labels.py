import argparse
import concurrent.futures as cf
import gzip
import multiprocessing as mp
import pickle
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heuristics.utils import (
    DUAL_OPTION_DEFAULT,
    UNIVERSAL_CUTOFFBOUND_DEFAULT,
    SCIPBranchingContext,
    compute_strong_branch_score,
    load_sample,
    resolve_effective_cutoffbound,
    unpack_sample_data,
)


LEGACY_TARGET_KEY = "top8_regression_targets"
TARGET_KEY_PREFIX = "top8_regression_targets_option"


def target_key_for_option(option: int) -> str:
    return f"{TARGET_KEY_PREFIX}{int(option)}"


def parse_dual_options(spec: str) -> list[int]:
    options = [int(x.strip()) for x in str(spec).split(",") if x.strip()]
    if len(options) == 0:
        raise ValueError("dual options list is empty")
    for option in options:
        if option not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported dual option {option}. Use values in {{1,2,3,4}}.")
    return options


def resolve_sample_files(data_path: str, split: str, max_samples: Optional[int]):
    dataset_root = Path(data_path)
    file_pattern = "sample_*.pkl"
    if split == "all":
        split_dirs = [
            dataset_root / "train",
            dataset_root / "valid",
            dataset_root / "test",
        ]
    elif split == "train":
        split_dirs = [dataset_root / "train"]
    elif split == "valid":
        split_dirs = [dataset_root / "valid"]
    elif split == "test":
        split_dirs = [dataset_root / "test"]
    else:
        raise ValueError(f"unsupported split: {split}")

    sample_files = []
    for split_dir in split_dirs:
        sample_files.extend(sorted(split_dir.glob(file_pattern)))
    if max_samples is not None:
        sample_files = sample_files[: int(max_samples)]
    return sample_files


def select_topk_local_positions(candidate_scores: np.ndarray, k: int):
    candidate_scores = torch.as_tensor(
        np.asarray(candidate_scores, dtype=np.float32).reshape(-1),
        dtype=torch.float32,
    )
    if int(candidate_scores.numel()) == 0:
        raise ValueError("No candidates available in sample action_set.")
    k_eff = min(int(k), int(candidate_scores.numel()))
    top_local = torch.topk(candidate_scores, k=k_eff).indices
    if k_eff < k:
        top_local = torch.cat([top_local, top_local[:1].expand(k - k_eff)], dim=0)
    return top_local


def sample_has_requested_targets(raw_sample: dict, dual_options: list[int]) -> bool:
    for dual_option in dual_options:
        option_target = raw_sample.get(target_key_for_option(int(dual_option)))
        if not isinstance(option_target, dict):
            return False
        if int(option_target.get("dual_option", -1)) != int(dual_option):
            return False
    return True


def build_targets_for_sample(
    raw_sample: dict,
    top_positions: np.ndarray,
    top_candidates: np.ndarray,
    top_scores: np.ndarray,
    dual_options: list[int],
    universal_cutoffbound: float,
    option_time_limit_sec: float,
    quick_test: bool = False,
    quick_test_preview_k: int = 8,
):
    top_positions = np.asarray(top_positions, dtype=np.int64)
    top_candidates = np.asarray(top_candidates, dtype=np.int64)
    top_scores = np.asarray(top_scores, dtype=np.float32)
    sample_record = unpack_sample_data(raw_sample["data"])
    context = SCIPBranchingContext.from_sample_state(sample_record.sample_state)
    parent_primal = context.solve_parent_primal()
    if (not parent_primal.success) or (parent_primal.objective_value is None):
        raise RuntimeError(f"Parent LP solve failed: {parent_primal.status} ({parent_primal.message})")
    cutoffbound = float(sample_record.cutoffbound)
    parent_obj = float(parent_primal.objective_value)
    n_cons = int(context.lp.A_ub.shape[0])
    n_vars = int(np.asarray(context.lp.objective_coefficients, dtype=np.float64).shape[0])
    k = int(top_candidates.shape[0])
    quick_logs = []

    if quick_test:
        preview = int(min(max(int(quick_test_preview_k), 1), k))
        quick_logs.append(f"[quick_test] parent_obj={parent_obj:.8g} sample_cutoffbound={cutoffbound:.8g} k={k}")
        quick_logs.append(f"[quick_test] top_candidates(first {preview})={top_candidates[:preview].tolist()}")

    for dual_option in dual_options:
        option_start_time = time.perf_counter()
        option_timed_out = False
        solved_down = 0
        solved_up = 0
        true_y = np.full((k, 2, n_cons), np.nan, dtype=np.float32)
        true_alpha = np.full((k, 2, n_vars), np.nan, dtype=np.float32)
        true_beta = np.full((k, 2, n_vars), np.nan, dtype=np.float32)
        true_obj = np.full((k, 2), np.nan, dtype=np.float32)

        for j, var_idx in enumerate(top_candidates.tolist()):
            remaining = float(option_time_limit_sec) - (time.perf_counter() - option_start_time)
            if remaining <= 0.0:
                option_timed_out = True
                break
            down = context.solve_child_dual(
                int(var_idx),
                direction="down",
                cutoffbound=cutoffbound,
                dual_option=int(dual_option),
                universal_cutoffbound=float(universal_cutoffbound),
                time_limit_sec=remaining,
            )
            if str(down.status).lower() in {"timelimit", "timelimit_reached"}:
                option_timed_out = True
            up = context.solve_child_dual(
                int(var_idx),
                direction="up",
                cutoffbound=cutoffbound,
                dual_option=int(dual_option),
                universal_cutoffbound=float(universal_cutoffbound),
                time_limit_sec=max(
                    1e-9,
                    float(option_time_limit_sec) - (time.perf_counter() - option_start_time),
                ),
            )
            if str(up.status).lower() in {"timelimit", "timelimit_reached"}:
                option_timed_out = True

            if down.success:
                true_y[j, 0, :] = down.y.astype(np.float32, copy=False)
                true_alpha[j, 0, :] = down.alpha.astype(np.float32, copy=False)
                true_beta[j, 0, :] = down.beta.astype(np.float32, copy=False)
                true_obj[j, 0] = np.float32(down.objective_value)
                solved_down += 1
            if up.success:
                true_y[j, 1, :] = up.y.astype(np.float32, copy=False)
                true_alpha[j, 1, :] = up.alpha.astype(np.float32, copy=False)
                true_beta[j, 1, :] = up.beta.astype(np.float32, copy=False)
                true_obj[j, 1] = np.float32(up.objective_value)
                solved_up += 1
            if option_timed_out:
                break

        effective_cutoffbound = float(
            resolve_effective_cutoffbound(
                cutoffbound=cutoffbound,
                dual_option=int(dual_option),
                universal_cutoffbound=float(universal_cutoffbound),
            )
        )

        if int(dual_option) == int(DUAL_OPTION_DEFAULT):
            option_scores = top_scores.copy()
        else:
            option_scores = np.full((k,), np.nan, dtype=np.float32)
            for j in range(k):
                down_obj = float(true_obj[j, 0])
                up_obj = float(true_obj[j, 1])
                if np.isfinite(down_obj) and np.isfinite(up_obj):
                    option_scores[j] = np.float32(
                        compute_strong_branch_score(
                            parent_obj=parent_obj,
                            child_one_obj=up_obj,
                            child_zero_obj=down_obj,
                            cutoffbound=effective_cutoffbound,
                        )
                    )

        option_target = {
            "candidate_positions": top_positions.copy(),
            "candidate_indices": top_candidates.copy(),
            "scores": option_scores,
            "sample_scores": top_scores.copy(),
            "y": true_y,
            "alpha": true_alpha,
            "beta": true_beta,
            "obj": true_obj,
            "parent_obj": np.float32(parent_obj),
            "cutoffbound": np.float32(effective_cutoffbound),
            "sample_cutoffbound": np.float32(cutoffbound),
            "dual_option": int(dual_option),
        }
        raw_sample[target_key_for_option(int(dual_option))] = option_target
        if int(dual_option) == int(DUAL_OPTION_DEFAULT):
            raw_sample[LEGACY_TARGET_KEY] = option_target

        if quick_test:
            down_finite = np.isfinite(true_obj[:, 0])
            up_finite = np.isfinite(true_obj[:, 1])
            valid_pairs = down_finite & up_finite
            preview = int(min(max(int(quick_test_preview_k), 1), k))
            obj_preview = [
                (float(true_obj[j, 0]), float(true_obj[j, 1]))
                for j in range(preview)
            ]
            score_preview = option_scores[:preview].tolist()
            quick_logs.append(
                f"[quick_test][option {int(dual_option)}] "
                f"effective_cutoffbound={effective_cutoffbound:.8g} "
                f"elapsed={time.perf_counter() - option_start_time:.2f}s "
                f"timeout={option_timed_out} "
                f"solved_down={solved_down}/{k} solved_up={solved_up}/{k} "
                f"down_finite={int(down_finite.sum())}/{k} "
                f"up_finite={int(up_finite.sum())}/{k} "
                f"valid_pairs={int(valid_pairs.sum())}/{k}"
            )
            quick_logs.append(f"[quick_test][option {int(dual_option)}] obj_preview={obj_preview}")
            quick_logs.append(f"[quick_test][option {int(dual_option)}] score_preview={score_preview}")

    return raw_sample, quick_logs


def process_sample_worker(task: tuple):
    (
        sample_path_str,
        top_positions,
        top_candidates,
        top_scores,
        dual_options,
        universal_cutoffbound,
        option_time_limit_sec,
        dry_run,
        quick_test,
        quick_test_preview_k,
    ) = task
    sample_path = Path(sample_path_str)
    try:
        raw_sample = load_sample(sample_path)
        raw_sample, quick_logs = build_targets_for_sample(
            raw_sample=raw_sample,
            top_positions=top_positions,
            top_candidates=top_candidates,
            top_scores=top_scores,
            dual_options=dual_options,
            universal_cutoffbound=float(universal_cutoffbound),
            option_time_limit_sec=float(option_time_limit_sec),
            quick_test=bool(quick_test),
            quick_test_preview_k=int(quick_test_preview_k),
        )
        if not bool(dry_run):
            with gzip.open(sample_path, "wb") as f:
                pickle.dump(raw_sample, f)
        return {"sample_path": str(sample_path), "quick_logs": quick_logs}
    except Exception as exc:
        raise RuntimeError(f"Failed to process {sample_path}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Generate top-8 regression targets and save into sample files.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "valid", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--dual_options",
        type=str,
        default="1,2,3,4",
        help="Comma-separated dual options to generate (values in {1,2,3,4}).",
    )
    parser.add_argument(
        "--universal_cutoffbound",
        type=float,
        default=UNIVERSAL_CUTOFFBOUND_DEFAULT,
        help="Universal cutoff used by dual options 2 and 4.",
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Run a one-sample sanity check with printed objective summaries.",
    )
    parser.add_argument(
        "--option_time_limit_sec",
        type=float,
        default=120.0,
        help="Per-option wall-clock time budget in seconds (applies across all k candidates).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for dual solve/write stage.",
    )
    args = parser.parse_args()
    quick_test_preview_k = 8
    dry_run = bool(args.quick_test)
    if args.quick_test:
        args.max_samples = 5
    num_workers = max(1, int(args.num_workers))
    if args.quick_test and num_workers > 1:
        print("[quick_test] forcing --num_workers=1 for deterministic logging.", flush=True)
        num_workers = 1
    dual_options = parse_dual_options(args.dual_options)

    sample_files = resolve_sample_files(args.data_path, args.split, args.max_samples)
    if len(sample_files) == 0:
        raise RuntimeError("no sample files found")

    resume_idx_reported = False
    n_written = 0
    n_skipped = 0
    progress_interval = 50
    next_progress_report = progress_interval
    executor = None
    pending_futures: dict[cf.Future, str] = {}
    max_in_flight = max(2, num_workers * 3)
    if num_workers > 1:
        executor = cf.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn"))

    try:
        for idx in range(len(sample_files)):
            raw_sample = load_sample(sample_files[idx])
            if sample_has_requested_targets(raw_sample, dual_options):
                n_skipped += 1
                if (n_written + n_skipped) >= next_progress_report:
                    print(
                        f"[progress] done={n_written + n_skipped}/{len(sample_files)} "
                        f"(written={n_written}, skipped={n_skipped}, in_flight={len(pending_futures)})",
                        flush=True,
                    )
                    while (n_written + n_skipped) >= next_progress_report:
                        next_progress_report += progress_interval
                continue
            if not resume_idx_reported:
                print(f"[resume] starting at {sample_files[idx].name}", flush=True)
                resume_idx_reported = True

            sample_record = unpack_sample_data(raw_sample["data"])
            top_local = select_topk_local_positions(sample_record.scores, int(args.k))
            top_local_np = top_local.detach().cpu().numpy().astype(np.int64, copy=False)
            task = (
                str(sample_files[idx]),
                top_local_np,
                sample_record.action_set[top_local_np].astype(np.int64, copy=False),
                sample_record.scores[top_local_np].astype(np.float32, copy=False),
                dual_options,
                float(args.universal_cutoffbound),
                float(args.option_time_limit_sec),
                bool(dry_run),
                bool(args.quick_test),
                int(quick_test_preview_k),
            )

            if executor is None:
                result = process_sample_worker(task)
                n_written += 1
                if args.quick_test:
                    print(f"[quick_test] sample={result['sample_path']}", flush=True)
                    for line in result["quick_logs"]:
                        print(line, flush=True)
                if (n_written + n_skipped) >= next_progress_report:
                    print(
                        f"[progress] done={n_written + n_skipped}/{len(sample_files)} "
                        f"(written={n_written}, skipped={n_skipped}, in_flight={len(pending_futures)})",
                        flush=True,
                    )
                    while (n_written + n_skipped) >= next_progress_report:
                        next_progress_report += progress_interval
            else:
                future = executor.submit(process_sample_worker, task)
                pending_futures[future] = str(sample_files[idx])
                if len(pending_futures) >= max_in_flight:
                    done, _ = cf.wait(pending_futures, return_when=cf.FIRST_COMPLETED)
                    for finished in done:
                        _ = finished.result()
                        pending_futures.pop(finished, None)
                        n_written += 1
                        if (n_written + n_skipped) >= next_progress_report:
                            print(
                                f"[progress] done={n_written + n_skipped}/{len(sample_files)} "
                                f"(written={n_written}, skipped={n_skipped}, in_flight={len(pending_futures)})",
                                flush=True,
                            )
                            while (n_written + n_skipped) >= next_progress_report:
                                next_progress_report += progress_interval

        for finished in cf.as_completed(list(pending_futures.keys())):
            _ = finished.result()
            n_written += 1
            if (n_written + n_skipped) >= next_progress_report:
                print(
                    f"[progress] done={n_written + n_skipped}/{len(sample_files)} "
                    f"(written={n_written}, skipped={n_skipped}, in_flight={len(pending_futures)})",
                    flush=True,
                )
                while (n_written + n_skipped) >= next_progress_report:
                    next_progress_report += progress_interval
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    options_str = ",".join(str(x) for x in dual_options)
    if dry_run:
        print(
            f"dry_run completed for {n_written} samples (skipped {n_skipped}; options: {options_str}).",
            flush=True,
        )
    else:
        print(
            f"saved {TARGET_KEY_PREFIX}* for {n_written} samples "
            f"(skipped {n_skipped}; options: {options_str}).",
            flush=True,
        )


if __name__ == "__main__":
    main()
