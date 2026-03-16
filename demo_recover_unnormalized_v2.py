#!/usr/bin/env python3
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import scip_data_helper_v2 as sdh
import utils


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


def _select_sample_path(dataset_path: Path, split: str, sample_index: int, eval_batch_size: int) -> Path:
    data = utils.load_data(_build_load_args(dataset_path, eval_batch_size), for_training=False)
    split_key = f"{split}_files"
    split_files = data.get(split_key, [])
    if not split_files:
        raise RuntimeError(f"No sample files found for split '{split}' under {dataset_path}")

    if sample_index < 0 or sample_index >= len(split_files):
        raise IndexError(f"sample_index {sample_index} out of range for split '{split}' (size={len(split_files)})")

    return Path(split_files[sample_index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact v2 demo: recover unnormalized LP and branch metrics with SCIP only")
    parser.add_argument("dataset_path", type=Path, help="Dataset directory with train/valid/test sample_*.pkl files")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=8, help="Top-k candidates by original score for SB reconstruction")
    parser.add_argument("--anchor_k", type=int, default=8, help="Number of anchors for dual-substitution demo")
    args = parser.parse_args()

    sample_path = _select_sample_path(args.dataset_path, args.split, args.sample_index, args.eval_batch_size)
    print(f"sample_file: {sample_path}")

    graph = utils.GraphDataset([sample_path], args=_build_graph_dataset_args()).get(0)
    print(
        "loaded_with_utils: "
        f"n_constraints={graph.n_constraints_per_graph} "
        f"n_variables={graph.n_variables_per_graph} "
        f"n_edges={graph.edge_index.shape[1]} "
        f"n_candidates={graph.nb_candidates}"
    )

    sample = utils.load_gzip(sample_path)
    record = sdh.unpack_sample_data(sample["data"])
    context = sdh.SCIPBranchingContext.from_sample_state(record.sample_state)

    obj_from_state = sdh.compute_primal_objective(context.lp)
    print(f"objective_from_solval_and_coef: {obj_from_state:.12g}")

    if context.lp.row_duals is not None and context.lp.reduced_costs is not None:
        obj_dual_rc = sdh.compute_dual_plus_reduced_cost_objective(context.lp)
        print(f"objective_from_dual_plus_rc:  {obj_dual_rc:.12g}")
        print(f"dual_rc_gap_vs_primal:        {abs(obj_dual_rc - obj_from_state):.3e}")
    else:
        print("objective_from_dual_plus_rc:  unavailable")

    parent = context.solve_parent_primal()
    print(f"parent_lp_status: {parent.status}")
    if parent.success and parent.objective_value is not None:
        print(f"objective_from_reconstructed_lp: {parent.objective_value:.12g}")
        print(f"solve_gap_vs_state_primal:      {abs(float(parent.objective_value) - obj_from_state):.3e}")

    print(f"expert_action_node: {record.action}")
    print(f"num_action_set: {len(record.action_set)}")
    print(f"num_scores: {len(record.scores)}")
    print(f"cutoffbound: {record.cutoffbound:.12g}")

    sb_topk = sdh.reconstruct_topk_strong_branching_scores(
        context=context,
        action_set=record.action_set,
        candidate_scores=record.scores,
        cutoffbound=record.cutoffbound,
        top_k=args.top_k,
    )
    print(f"reconstructed_strong_branching_topk: {sb_topk['k']}")
    for row in sb_topk["topk_by_score"]:
        print(
            "sb_topk_row: "
            f"cand_pos={row['cand_position']} "
            f"lp_pos={row['cand_lp_pos']} "
            f"orig_score={row['cand_score']:.12g} "
            f"reconstructed_score={row['computed_score']:.12g} "
            f"down_status={row['down_status']} "
            f"up_status={row['up_status']}"
        )
    print(f"sb_rank_original: {sb_topk['topk_rank_original']}")
    print(f"sb_rank_reconstructed: {sb_topk['topk_rank_computed']}")
    print(f"sb_rank_match: {sb_topk['topk_rank_original'] == sb_topk['topk_rank_computed']}")

    anchor_res = sdh.run_anchor_dual_substitution_with_scip(
        context=context,
        action_set=record.action_set,
        candidate_scores=record.scores,
        cutoffbound=record.cutoffbound,
        anchor_k=args.anchor_k,
    )
    print(f"anchor_k_eff: {len(anchor_res['anchor_positions'])}")
    print(f"anchor_positions: {anchor_res['anchor_positions'].tolist()}")
    print(f"anchor_dual_pool_size: {len(anchor_res['dual_pool_records'])}")

    for row in anchor_res["anchor_child_records"]:
        print(
            "anchor_child_row: "
            f"cand_pos={row['cand_position']} "
            f"lp_pos={row['cand_lp_pos']} "
            f"down_status={row['down_status']} down_obj={row['down_obj']:.12g} "
            f"down_dual_status={row['down_dual_status']} "
            f"down_dual_eval_gap={row['down_dual_eval_gap']} "
            f"up_status={row['up_status']} up_obj={row['up_obj']:.12g} "
            f"up_dual_status={row['up_dual_status']} "
            f"up_dual_eval_gap={row['up_dual_eval_gap']}"
        )

    pseudo_scores = anchor_res["pseudo_scores"]
    top_pseudo_positions = np.argsort(pseudo_scores)[-args.top_k:][::-1]
    print(f"anchor_selected_position: {anchor_res['selected_position']}")
    print(f"anchor_selected_lp_pos: {anchor_res['selected_lp_pos']}")
    print(f"anchor_top{args.top_k}_by_pseudo:")
    for pos in top_pseudo_positions.tolist():
        print(
            "anchor_pseudo_row: "
            f"cand_pos={pos} "
            f"lp_pos={int(anchor_res['candidate_lp_positions'][pos])} "
            f"orig_score={float(anchor_res['candidate_scores'][pos]):.12g} "
            f"pseudo_score={float(anchor_res['pseudo_scores'][pos]):.12g}"
        )


if __name__ == "__main__":
    main()
