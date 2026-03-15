#!/usr/bin/env python3
import argparse
from pathlib import Path
from types import SimpleNamespace

import utils
import scip_data_helper as sdh


def _build_load_args(dataset_path: Path, eval_batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_path=str(dataset_path),
        eval_batch_size=eval_batch_size,
        train_split="train",
        val_split="valid",
        test_split="test",
        file_pattern="sample_*.pkl",
        edge_nfeats=1,
        two_fwl=False,
    )


def _build_graph_dataset_args() -> SimpleNamespace:
    return SimpleNamespace(
        use_default_features=True,
        remove_bad_candidates=False,
        relevance_type="linear",
        tier1_ub=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: recover unnormalized SCIP features from saved samples.")
    parser.add_argument("dataset_path", type=Path, help="Dataset directory with train/valid/test sample_*.pkl files")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--solve_reconstructed_lp",
        action="store_true",
        help="Solve reconstructed LP and compare objective with recovered state objective.",
    )
    parser.add_argument(
        "--solver",
        choices=["scip"],
        default="scip",
        help="Solver backend for reconstructed LP (default: scip).",
    )
    parser.add_argument(
        "--scip_as_mip",
        action="store_true",
        help="When --solver scip is used, solve as MIP using recovered variable types (default: LP relaxation).",
    )
    parser.add_argument(
        "--no_assume_binary_bounds",
        action="store_true",
        help="Disable fallback binary bounds [0,1] when explicit lb/ub metadata is unavailable.",
    )
    parser.add_argument(
        "--strong_branch_top_k",
        type=int,
        default=8,
        help="Top-k candidates (by original score) for strong-branch score reconstruction.",
    )
    args = parser.parse_args()

    data = utils.load_data(_build_load_args(args.dataset_path, args.eval_batch_size), for_training=False)
    split_key = f"{args.split}_files"
    split_files = data.get(split_key, [])
    if not split_files:
        raise RuntimeError(f"No sample files found for split '{args.split}' under {args.dataset_path}")

    if args.sample_index < 0 or args.sample_index >= len(split_files):
        raise IndexError(
            f"sample_index {args.sample_index} out of range for split '{args.split}' (size={len(split_files)})"
        )

    sample_path = Path(split_files[args.sample_index])
    print(f"sample_file: {sample_path}")

    # 1) Load via utils.py exactly as training code does.
    graph = utils.GraphDataset([sample_path], args=_build_graph_dataset_args()).get(0)
    print(
        "loaded_with_utils: "
        f"n_constraints={graph.n_constraints_per_graph}, "
        f"n_variables={graph.n_variables_per_graph}, "
        f"n_edges={graph.edge_index.shape[1]}, "
        f"n_candidates={graph.nb_candidates}"
    )

    # 2) Recover unnormalized features from raw saved state.
    sample = utils.load_gzip(sample_path)
    sample_state, sample_action, sample_action_set, sample_scores, cutoffbound = sdh.unpack_sample_data(sample["data"])
    unnormalized_state = sdh.reconstruct_unnormalized_state(sample_state)

    lp = sdh.extract_lp_components(
        unnormalized_state,
        assume_binary_bounds=(not args.no_assume_binary_bounds),
    )

    # 3) Objective checks aligned with 02_generate_samples.py logic:
    #    primal: c^T x (+ offset)
    #    dual+rc: row_duals^T row_bias + redcosts^T solvals
    obj_primal = sdh.compute_primal_objective(lp)
    print(f"objective_from_solval_and_coef: {obj_primal:.12g}")

    if lp["row_duals"] is not None and lp["reduced_costs"] is not None:
        obj_dual_rc = sdh.compute_dual_reduced_cost_objective(lp)
        print(f"objective_from_dual_plus_rc:  {obj_dual_rc:.12g}")
        print(f"dual_rc_gap_vs_primal:        {abs(obj_dual_rc - obj_primal):.3e}")
    else:
        print("objective_from_dual_plus_rc:  unavailable (missing duals or reduced costs in features)")

    print(f"expert_action_node: {sample_action}")
    print(f"num_action_set: {len(sample_action_set)}")
    print(f"num_scores: {len(sample_scores)}")
    print(f"cutoffbound: {cutoffbound:.12g}")

    sb_topk = sdh.reconstruct_topk_strong_branching_scores(
        lp_components=lp,
        action_set=sample_action_set,
        candidate_scores=sample_scores,
        cutoffbound=cutoffbound,
        top_k=args.strong_branch_top_k,
        as_mip=args.scip_as_mip,
    )
    print(f"reconstructed_strong_branching_topk: {sb_topk['k']}")
    for row in sb_topk["topk_by_score"]:
        print(
            "sb_topk_row: "
            f"cand_pos={row['cand_position']} "
            f"lp_pos={row['cand_lp_pos']} "
            f"orig_score={row['cand_score']:.12g} "
            f"reconstructed_score={row['computed_score']:.12g} "
            f"parent_obj={row['parent_lp_obj']:.12g} "
            f"child1_obj={row['child_one_lp_obj']:.12g} "
            f"child0_obj={row['child_zero_lp_obj']:.12g} "
            f"down_status={row['down_status']} "
            f"up_status={row['up_status']}"
        )
    print(f"sb_rank_original: {sb_topk['topk_rank_original']}")
    print(f"sb_rank_reconstructed: {sb_topk['topk_rank_computed']}")
    print(f"sb_rank_match: {sb_topk['topk_rank_original'] == sb_topk['topk_rank_computed']}")

    if args.solve_reconstructed_lp:
        if args.solver == "scip":
            result = sdh.solve_reconstructed_lp_with_scip(lp, as_mip=args.scip_as_mip)
        else:
            raise NotImplementedError
        print(f"reconstructed_lp_solved: {result['success']}")
        print(f"reconstructed_lp_status: {result['status']} ({result['message']})")
        if result["success"]:
            obj_solved = float(result["objective_value"])
            print(f"objective_from_reconstructed_lp: {obj_solved:.12g}")
            print(f"solve_gap_vs_state_primal:      {abs(obj_solved - obj_primal):.3e}")


if __name__ == "__main__":
    main()
