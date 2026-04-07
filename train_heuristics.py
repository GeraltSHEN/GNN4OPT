import time
import argparse
import os
from pathlib import Path
import pdb
from typing import Any, Dict, Optional, Sequence

try:
    import psutil
except ImportError:
    psutil = None
import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from tmp_utils import (
    get_optimizer,
    load_model,
    load_data,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    print_dash_str
)
from heuristics.postprocess_interface import (
    HeuristicPostProcessInterface,
    IndexedGraphDataset,
    load_sample,
)
from losses import (
    NormalizedPairwiseLogisticLoss,
    TierNormalizedLambdaARP2,
    LiPO,
    TopTierAverageSoftmaxLoss,
    TierAwarePairwiseLogisticLoss,
    NCE
)
from pytorchltr.loss import LambdaNDCGLoss1, LambdaNDCGLoss2, LambdaARPLoss1, LambdaARPLoss2, PairwiseLogisticLoss

def log_cpu_memory_usage(epoch: int, step: Optional[str] = None):
    """Report CPU memory usage at coarse intervals."""
    if psutil is None:
        return
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    if epoch == 1 or epoch % 1 == 0:
        tag = f", Step {step}" if step is not None else ""
        print(
            f"[Epoch {epoch}{tag}] CPU Memory - RSS: {memory_info.rss / (1024 ** 3):.2f} GB"
        )


def _infer_feature_dimensions(train_loader):
    """Infer feature dimensions from a single sample in the training dataset."""
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None or len(dataset) == 0:
        raise ValueError("Training dataset is empty; cannot infer feature dimensions.")
    sample = dataset[0]
    cons_nfeats = sample.constraint_features.shape[-1]
    edge_nfeats = sample.edge_attr.shape[-1]
    var_nfeats = sample.variable_features.shape[-1]
    return cons_nfeats, edge_nfeats, var_nfeats


_BATCH_POSTPROCESS_KEYS = (
    "post_rhs",
    "post_parent_lbs",
    "post_parent_ubs",
    "post_lp_solution",
    "post_parent_obj",
    "post_cutoffbound",
    "post_objective_offset",
)


def _build_batch_postprocess_data(batch):
    if not all(hasattr(batch, key) for key in _BATCH_POSTPROCESS_KEYS):
        return None
    device = batch.constraint_features.device
    dtype = batch.constraint_features.dtype
    return {
        "rhs": batch.post_rhs.to(device=device, dtype=dtype),
        "parent_lbs": batch.post_parent_lbs.to(device=device, dtype=dtype),
        "parent_ubs": batch.post_parent_ubs.to(device=device, dtype=dtype),
        "lp_solution": batch.post_lp_solution.to(device=device, dtype=dtype),
        "parent_obj": batch.post_parent_obj.reshape(-1).to(device=device, dtype=dtype),
        "cutoffbound": batch.post_cutoffbound.reshape(-1).to(device=device, dtype=dtype),
        "objective_offset": batch.post_objective_offset.reshape(-1).to(device=device, dtype=dtype),
    }


def _extract_topk_targets_from_batch(batch):
    if batch is None or not hasattr(batch, "topk_targets"):
        return None
    batched_targets = batch.topk_targets
    if not isinstance(batched_targets, dict):
        return None

    batch_size = int(getattr(batch, "num_graphs", 0))
    if batch_size <= 0:
        return None

    targets = []
    for idx in range(batch_size):
        sample_target = {}
        for key, value in batched_targets.items():
            if isinstance(value, list):
                sample_target[key] = value[idx]
            elif torch.is_tensor(value):
                sample_target[key] = value[idx]
            else:
                try:
                    sample_target[key] = value[idx]
                except Exception:
                    sample_target[key] = value
        targets.append(sample_target)
    return targets


def _forward_with_optional_postprocess(
    policy,
    batch,
    postprocess_interface=None,
    *,
    return_aux: bool = False,
):
    if hasattr(policy, "post_process"):
        post_data = _build_batch_postprocess_data(batch)
        if post_data is None:
            if postprocess_interface is None:
                raise ValueError(
                    "Post-process data not found on batch and postprocess_interface is None."
                )
            post_data = postprocess_interface.make_batch_data(
                batch.graph_id,
                device=batch.constraint_features.device,
                dtype=batch.constraint_features.dtype,
            )
        if return_aux:
            logits, padded_logits, aux = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                candidates=batch.candidates,
                nb_candidates=batch.nb_candidates,
                n_constraints_per_graph=batch.n_constraints_per_graph,
                n_variables_per_graph=batch.n_variables_per_graph,
                data=post_data,
                return_aux=True,
            )
            if isinstance(aux, dict):
                aux = {**aux, "post_data": post_data}
            else:
                aux = {"model_aux": aux, "post_data": post_data}
            return logits, padded_logits, aux
        logits, padded_logits, _ = policy(
            batch.constraint_features,
            batch.edge_index,
            batch.edge_attr,
            batch.variable_features,
            candidates=batch.candidates,
            nb_candidates=batch.nb_candidates,
            n_constraints_per_graph=batch.n_constraints_per_graph,
            n_variables_per_graph=batch.n_variables_per_graph,
            data=post_data,
        )
        return logits, padded_logits

    logits = policy(
        batch.constraint_features,
        batch.edge_index,
        batch.edge_attr,
        batch.variable_features,
        candidates=batch.candidates,
        n_constraints_per_graph=batch.n_constraints_per_graph,
        n_variables_per_graph=batch.n_variables_per_graph,
    )
    if return_aux:
        return logits, None, None
    return logits, None


def _wrap_loader_for_postprocess(
    loader,
    *,
    shuffle: bool,
    dual_option: int = 1,
    universal_cutoffbound: float = 1e6,
    max_cache_size: int = 128,
):
    if loader is None:
        return None, None

    indexed_dataset = IndexedGraphDataset(loader.dataset)
    loader_kwargs = dict(
        dataset=indexed_dataset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        persistent_workers=getattr(loader, "persistent_workers", False),
    )
    prefetch_factor = getattr(loader, "prefetch_factor", None)
    if loader.num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    wrapped_loader = DataLoader(**loader_kwargs)
    postprocess_interface = HeuristicPostProcessInterface(
        indexed_dataset.sample_files,
        dual_option=int(dual_option),
        universal_cutoffbound=float(universal_cutoffbound),
        max_cache_size=int(max_cache_size),
    )
    return wrapped_loader, postprocess_interface


TOPK_TARGET_KEY = "top8_regression_targets"
TOPK_TARGET_KEY_PREFIX = "top8_regression_targets_option"
GRAD_MONITOR_PARAM_NAMES = (
    "var_embedding.1.lins.0.weight",
    "data_encoder.conv_0_v_to_c.output_module.2.weight",
    "data_encoder.conv_0_c_to_v.output_module.2.weight",
    "vars_out.1.lins.0.weight",
    "vars_out.1.lins.1.weight",
)


def _topk_target_key_for_option(dual_option: int) -> str:
    return f"{TOPK_TARGET_KEY_PREFIX}{int(dual_option)}"


def _tensor_norm_or_zero(tensor: Optional[torch.Tensor]) -> float:
    if tensor is None:
        return 0.0
    return float(tensor.norm().item())


def _log_gradient_diagnostics(
    writer: SummaryWriter,
    policy: torch.nn.Module,
    step: int,
):
    named_params = dict(policy.named_parameters())

    total_grad_sq = 0.0
    for param in policy.parameters():
        if param.grad is None:
            continue
        grad_norm = float(param.grad.norm().item())
        total_grad_sq += grad_norm * grad_norm
    writer.add_scalar("GradNorm/total", total_grad_sq ** 0.5, step)

    for name in GRAD_MONITOR_PARAM_NAMES:
        param = named_params.get(name)
        if param is None:
            continue
        writer.add_scalar(f"GradNorm/{name}", _tensor_norm_or_zero(param.grad), step)
        writer.add_scalar(f"ParamNorm/{name}", _tensor_norm_or_zero(param.data), step)

    var_embed_weight = named_params.get("var_embedding.1.lins.0.weight")
    if var_embed_weight is not None and var_embed_weight.dim() == 2 and var_embed_weight.size(1) >= 3:
        param_base = var_embed_weight.data[:, :-2]
        param_branch = var_embed_weight.data[:, -2:]
        param_base_norm = _tensor_norm_or_zero(param_base)
        param_branch_norm = _tensor_norm_or_zero(param_branch)
        writer.add_scalar("ParamNorm/var_embedding_l0_base", param_base_norm, step)
        writer.add_scalar("ParamNorm/var_embedding_l0_branch", param_branch_norm, step)
        writer.add_scalar(
            "ParamRatio/var_embedding_l0_branch_over_base",
            param_branch_norm / max(param_base_norm, 1e-12),
            step,
        )

        if var_embed_weight.grad is not None:
            grad_base = var_embed_weight.grad[:, :-2]
            grad_branch = var_embed_weight.grad[:, -2:]
            grad_base_norm = _tensor_norm_or_zero(grad_base)
            grad_branch_norm = _tensor_norm_or_zero(grad_branch)
            writer.add_scalar("GradNorm/var_embedding_l0_base", grad_base_norm, step)
            writer.add_scalar("GradNorm/var_embedding_l0_branch", grad_branch_norm, step)
            writer.add_scalar(
                "GradRatio/var_embedding_l0_branch_over_base",
                grad_branch_norm / max(grad_base_norm, 1e-12),
                step,
            )


def _load_saved_topk_targets(
    graph_id: torch.Tensor,
    postprocess_interface,
    target_keys: Sequence[str],
    batch=None,
):
    batch_targets = _extract_topk_targets_from_batch(batch)
    if batch_targets is not None:
        return batch_targets
    if postprocess_interface is None:
        raise ValueError(
            "Unable to load top-k targets: missing both batched topk_targets and postprocess interface."
        )
    if graph_id is None:
        raise ValueError(
            "Unable to load top-k targets from files: graph_id is missing on the batch."
        )
    targets = []
    for gid in graph_id.reshape(-1).detach().cpu().tolist():
        sample = load_sample(postprocess_interface.sample_files[int(gid)])
        selected = None
        for key in target_keys:
            if key in sample:
                selected = sample[key]
                break
        if selected is None:
            raise KeyError(
                f"Missing regression target keys {list(target_keys)} in sample {postprocess_interface.sample_files[int(gid)]}"
            )
        targets.append(selected)
    return targets


def _assert_topk_target_alignment(model_aux: Dict[str, torch.Tensor], saved_targets: Sequence[Dict[str, Any]]):
    top_local = model_aux.get("top_local")
    top_global = model_aux.get("branching_candidates_global")
    if top_local is None or top_global is None:
        return
    top_local = top_local.to(dtype=torch.long)
    top_global = top_global.to(dtype=torch.long)
    for b_idx, target in enumerate(saved_targets):
        target_local = torch.as_tensor(target["candidate_positions"], device=top_local.device, dtype=torch.long)
        target_global = torch.as_tensor(target["candidate_indices"], device=top_global.device, dtype=torch.long)
        assert torch.equal(top_local[b_idx], target_local), "Top-k local order mismatch with saved targets."
        assert torch.equal(top_global[b_idx], target_global), "Top-k candidate ids mismatch with saved targets."


def train(
    args,
    policy,
    optimizer,
    train_dataloader,
    *,
    start_step: int = 0,
    model_dir: Path,
    log_dir: Path,
    val_dataloader=None,
    train_postprocess_interface=None,
    val_postprocess_interface=None,
):
    policy.train()
    device = args.device
    epochs = args.epochs
    eval_every = args.eval_every
    save_every = args.save_every
    print_every = args.print_every
    loss_option = args.loss_option
    regression_target = None
    include_parent = bool(getattr(args, "include_parent", False))
    dual_option = int(getattr(args, "dual_option", 1))
    if dual_option not in (1, 2):
        raise ValueError(f"Unsupported dual_option '{dual_option}'. Use one of: 1, 2.")
    if dual_option == 1:
        topk_target_keys = [
            _topk_target_key_for_option(1),
            TOPK_TARGET_KEY,
            _topk_target_key_for_option(2),
        ]
    else:
        topk_target_keys = [
            _topk_target_key_for_option(2),
            _topk_target_key_for_option(1),
            TOPK_TARGET_KEY,
        ]
    if loss_option == "regression":
        regression_target = str(getattr(args, "regression_target", "score")).lower()
    ranking_loss_factories = {
        "LambdaNDCGLoss1": LambdaNDCGLoss1,
        "LambdaNDCGLoss2": LambdaNDCGLoss2,
        "LambdaARPLoss1": LambdaARPLoss1,
        "LambdaARPLoss2": LambdaARPLoss2,
        "PairwiseLogisticLoss": PairwiseLogisticLoss,
        "NormalizedPairwiseLogisticLoss": NormalizedPairwiseLogisticLoss,
        "TierAwarePairwiseLogisticLoss": TierAwarePairwiseLogisticLoss,
        "TierNormalizedLambdaARP2": TierNormalizedLambdaARP2,
        "LiPO": LiPO,
        "TopTierAverageSoftmaxLoss": TopTierAverageSoftmaxLoss,
        "NCE": NCE,
    }
    ranking_loss_cls = ranking_loss_factories.get(loss_option)
    if ranking_loss_cls is None:
        ranking_loss_fn = None
    elif loss_option == "TierAwarePairwiseLogisticLoss":
        ranking_loss_fn = ranking_loss_cls(
            c_11=getattr(args, "c_11", 0.3),
            c_12=getattr(args, "c_12", 0.3),
            c_21=getattr(args, "c_21", 0.3),
            c_22=getattr(args, "c_22", 0.1),
        )
    else:
        ranking_loss_fn = ranking_loss_cls()
    score_th = float('inf')
    train_tb_every = max(int(getattr(args, "train_tb_every", 100)), 1)
    grad_log_every = max(int(getattr(args, "grad_log_every", train_tb_every)), 1)

    model_dir = Path(model_dir)
    log_dir = Path(log_dir)
    writer = SummaryWriter(log_dir=str(log_dir))

    num_gradient_steps = start_step
    for epoch in range(epochs):
        log_cpu_memory_usage(epoch + 1)
        mean_loss = 0
        mean_acc = 0
        mean_top5_acc = 0
        mean_score_diff = 0
        mean_normalized_score_diff = 0

        n_samples_processed = 0
        for batch in train_dataloader:
            if (val_dataloader is not None
                and eval_every
                and num_gradient_steps % eval_every == 0
            ):
                print_dash_str(
                    f"Evaluating at epoch {epoch + 1}, step {num_gradient_steps}"
                )
                (
                    valid_loss,
                    valid_acc,
                    valid_top5_acc,
                    valid_score_diff,
                    valid_normalized_score_diff,
                ) = evaluate(
                    policy,
                    val_dataloader,
                    device,
                    writer,
                    num_gradient_steps,
                    postprocess_interface=val_postprocess_interface,
                )
                print_dash_str(
                    (
                        f"Valid loss: {valid_loss:.3f}, accuracy {valid_acc:.3f}, "
                        f"top 5 accuracy {valid_top5_acc:.3f}, score difference [abs] {valid_score_diff:.3f} "
                        f"[relative] {valid_normalized_score_diff:.3f}"
                    )
                )

            if save_every and num_gradient_steps % save_every == 0:
                save_checkpoint(
                    policy, num_gradient_steps, optimizer, save_dir=str(model_dir)
                )

            if (
                print_every
                and num_gradient_steps % print_every == 0
                and n_samples_processed > 0
            ):
                print(
                    f"Step {num_gradient_steps}: Train loss {mean_loss / n_samples_processed:.3f}, "
                    f"accuracy {mean_acc / n_samples_processed:.3f}, "
                    f"Top-5 accuracy {mean_top5_acc / n_samples_processed:.3f}, "
                    f"[absolute] {mean_score_diff / n_samples_processed:.3f} "
                    f"[relative] {mean_normalized_score_diff / n_samples_processed:.3f}"
                )

            batch = batch.to(device, non_blocking=True)
            model_aux = None
            parent_loss_item = None
            if loss_option == "regression":
                flat_logits, precomputed_padded_logits, model_aux = _forward_with_optional_postprocess(
                    policy,
                    batch,
                    postprocess_interface=train_postprocess_interface,
                    return_aux=True,
                )
            else:
                flat_logits, precomputed_padded_logits = _forward_with_optional_postprocess(
                    policy,
                    batch,
                    postprocess_interface=train_postprocess_interface,
                )

            if score_th < float("inf"):
                select_indices = (
                    batch.candidate_scores.max(axis=-1).values < score_th
                )
                flat_logits = flat_logits[select_indices]
                batch = batch[select_indices]
                if len(flat_logits) == 0:
                    continue
            """
            train_heuristics loss options result in different operations, invovling some post-process operations to 
            HeuristicPolicy's outputs, unfinished instruction ...
            """
            if loss_option == "classification":
                logits = (
                    precomputed_padded_logits
                    if precomputed_padded_logits is not None
                    else pad_tensor(flat_logits[batch.candidates], batch.nb_candidates)
                )
                loss = F.cross_entropy(logits, batch.candidate_choices)
                nan_mask = torch.isnan(loss)
                if nan_mask.any():
                    raise ValueError("stop here")
            elif loss_option == "regression":
                logits = (
                    precomputed_padded_logits
                    if precomputed_padded_logits is not None
                    else pad_tensor(flat_logits[batch.candidates], batch.nb_candidates)
                )
                if model_aux is None:
                    target_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
                    loss = F.mse_loss(logits, target_scores)
                else:
                    saved_targets = _load_saved_topk_targets(
                        getattr(batch, "graph_id", None),
                        train_postprocess_interface,
                        target_keys=topk_target_keys,
                        batch=batch,
                    )
                    _assert_topk_target_alignment(model_aux, saved_targets)
                    if regression_target == "score":
                        true_scores = torch.stack(
                            [
                                torch.as_tensor(
                                    target["scores"],
                                    device=device,
                                    dtype=model_aux["topk_pseudo_scores"].dtype,
                                )
                                for target in saved_targets
                            ],
                            dim=0,
                        )
                        loss = F.mse_loss(model_aux["topk_pseudo_scores"], true_scores)
                    elif regression_target == "obj":
                        pred_obj = torch.stack(
                            [model_aux["topk_down_obj"], model_aux["topk_up_obj"]],
                            dim=-1,
                        )
                        true_obj = torch.stack(
                            [
                                torch.as_tensor(
                                    target["obj"],
                                    device=device,
                                    dtype=pred_obj.dtype,
                                )
                                for target in saved_targets
                            ],
                            dim=0,
                        )
                        obj_loss = F.mse_loss(pred_obj, true_obj)
                        if include_parent:
                            pred_parent_obj = model_aux["parent_obj"]
                            true_parent_obj = torch.stack(
                                [
                                    torch.as_tensor(
                                        target["parent_obj"],
                                        device=device,
                                        dtype=pred_parent_obj.dtype,
                                    )
                                    for target in saved_targets
                                ],
                                dim=0,
                            ).reshape(-1)
                            parent_loss = F.mse_loss(pred_parent_obj.reshape(-1), true_parent_obj)
                            parent_loss_item = float(parent_loss.detach().item())
                            loss = obj_loss + parent_loss
                        else:
                            loss = obj_loss
                    elif regression_target == "dual":
                        pred_y = model_aux["topk_y"]
                        pred_alpha = model_aux["topk_alpha"]
                        pred_beta = model_aux["topk_beta"]

                        true_y = torch.zeros_like(pred_y)
                        true_alpha = torch.zeros_like(pred_alpha)
                        true_beta = torch.zeros_like(pred_beta)
                        for b_idx, target in enumerate(saved_targets):
                            y_target = torch.as_tensor(target["y"], device=device, dtype=pred_y.dtype)
                            alpha_target = torch.as_tensor(
                                target["alpha"], device=device, dtype=pred_alpha.dtype
                            )
                            beta_target = torch.as_tensor(
                                target["beta"], device=device, dtype=pred_beta.dtype
                            )
                            true_y[b_idx, :, :, : y_target.size(-1)] = y_target
                            true_alpha[b_idx, :, :, : alpha_target.size(-1)] = alpha_target
                            true_beta[b_idx, :, :, : beta_target.size(-1)] = beta_target

                        y_mask = model_aux["real_y_mask_topk"].to(dtype=pred_y.dtype)
                        x_mask = model_aux["real_x_mask_topk"].to(dtype=pred_alpha.dtype)
                        loss = (
                            F.mse_loss(pred_y * y_mask, true_y * y_mask)
                            + F.mse_loss(pred_alpha * x_mask, true_alpha * x_mask)
                            + F.mse_loss(pred_beta * x_mask, true_beta * x_mask)
                        )
                    else:
                        raise ValueError(
                            f"Unsupported regression_target '{regression_target}'. Use one of: dual, obj, score."
                        )
            elif ranking_loss_fn is not None:
                logits = pad_tensor(flat_logits[batch.candidates], batch.nb_candidates,
                                    pad_value=0)
                loss_fn = ranking_loss_fn
                padded_relevance = pad_tensor(batch.candidate_relevance, batch.nb_candidates, pad_value=0)
                loss = loss_fn(logits, padded_relevance, batch.nb_candidates)
                nan_mask = torch.isnan(loss)
                if nan_mask.any():
                    nan_indices = nan_mask.nonzero(as_tuple=False).flatten()
                    print(f"NaN ranking loss at indices {nan_indices.tolist()}")
                    print(f"logits: {logits[nan_indices].detach().cpu()}")
                    print(f"padded_relevance: {padded_relevance[nan_indices].detach().cpu()}")
                    raise ValueError("stop here")
                loss = loss.mean()
            else:
                raise ValueError(f"Unsupported loss option: {loss_option}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if num_gradient_steps % grad_log_every == 0:
                _log_gradient_diagnostics(writer, policy, num_gradient_steps)
            optimizer.step()
            num_gradient_steps += 1

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates).clip(0)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()
            top5_acc = (true_scores.gather(-1, logits.topk(min(5, logits.size(-1))).indices) == true_bestscore).float().max(dim=-1).values.mean().item()
            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            mean_top5_acc += top5_acc * batch.num_graphs
            n_samples_processed += batch.num_graphs

            # torch.save(policy.state_dict(), "trained_params.pkl")
            score_diff = (true_bestscore - true_scores.gather(-1, predicted_bestindex).clip(0)).mean().item()
            normalized_score_diff = ((true_bestscore - true_scores.gather(-1, predicted_bestindex).clip(0)) / true_bestscore).mean().item()
            mean_score_diff += score_diff * batch.num_graphs
            mean_normalized_score_diff += normalized_score_diff * batch.num_graphs

            if num_gradient_steps % train_tb_every == 0:
                writer.add_scalar("Loss/train", loss.item(), num_gradient_steps)
                if parent_loss_item is not None:
                    writer.add_scalar("ParentObjLoss/train", parent_loss_item, num_gradient_steps)
                writer.add_scalar("Accuracy/train", accuracy, num_gradient_steps)
                writer.add_scalar("Top5_Accuracy/train", top5_acc, num_gradient_steps)
                writer.add_scalar("Score_diff/train", score_diff, num_gradient_steps)
                writer.add_scalar("Normalized_score_diff/train", normalized_score_diff, num_gradient_steps)

        if n_samples_processed == 0:
            print_dash_str(f"No samples processed in epoch {epoch + 1}.")
            continue

        mean_loss /= n_samples_processed
        mean_acc /= n_samples_processed
        mean_top5_acc /= n_samples_processed
        mean_score_diff /= n_samples_processed
        mean_normalized_score_diff /= n_samples_processed
        print(
            f"Epoch {epoch + 1}: Train loss {mean_loss:.3f}, accuracy {mean_acc:.3f}, "
            f"top 5 accuracy {mean_top5_acc:.3f}, [absolute] {mean_score_diff:.3f} "
            f"[relative] {mean_normalized_score_diff:.3f}"
        )

        writer.add_scalar("Loss/Epoch_train", mean_loss, epoch)
        writer.add_scalar("Accuracy/Epoch_train", mean_acc, epoch)
        writer.add_scalar("Top5_Accuracy/Epoch_train", mean_top5_acc, epoch)
        writer.add_scalar("Score_diff/Epoch_train", mean_score_diff, epoch)
        writer.add_scalar("Normalized_score_diff/Epoch_train", mean_normalized_score_diff, epoch)

    save_checkpoint(policy, num_gradient_steps, optimizer, save_dir=str(model_dir))
    writer.close()


def evaluate(policy, data_loader, device, writer, num_gradient_steps, postprocess_interface=None):
    mean_loss = 0
    mean_acc = 0
    mean_top5_acc = 0
    mean_score_diff = 0
    mean_normalized_score_diff = 0

    policy.eval()

    n_samples_processed = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, disable=True):
            batch = batch.to(device, non_blocking=True)
            flat_logits, precomputed_padded_logits = _forward_with_optional_postprocess(
                policy,
                batch,
                postprocess_interface=postprocess_interface,
            )
            logits = (
                precomputed_padded_logits
                if precomputed_padded_logits is not None
                else pad_tensor(flat_logits[batch.candidates], batch.nb_candidates)
            )
            # Compute the usual cross-entropy classification loss
            loss = F.cross_entropy(logits, batch.candidate_choices)
            # if isnan: pdb
            if torch.isnan(loss):
                pdb.set_trace()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates).clip(0)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()
            top5_acc = (true_scores.gather(-1, logits.topk(min(5, logits.size(-1))).indices) == true_bestscore).float().max(dim=-1).values.mean().item()

            score_diff = (true_bestscore - true_scores.gather(-1, predicted_bestindex)).abs().mean().item()
            normalized_score_diff = ((true_bestscore - true_scores.gather(-1, predicted_bestindex)) / true_bestscore).mean().item()

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            mean_top5_acc += top5_acc * batch.num_graphs

            mean_score_diff += score_diff * batch.num_graphs
            mean_normalized_score_diff += normalized_score_diff * batch.num_graphs
            n_samples_processed += batch.num_graphs

    if n_samples_processed > 0:
        mean_loss /= n_samples_processed
        mean_acc /= n_samples_processed
        mean_top5_acc /= n_samples_processed
        mean_score_diff /= n_samples_processed
        mean_normalized_score_diff /= n_samples_processed

    writer.add_scalar("Loss/val", mean_loss, num_gradient_steps)
    writer.add_scalar("Accuracy/val", mean_acc, num_gradient_steps)
    writer.add_scalar("Top5_Accuracy/val", mean_top5_acc, num_gradient_steps)
    writer.add_scalar("Score_diff/val", mean_score_diff, num_gradient_steps)
    writer.add_scalar("Normalized_score_diff/val", mean_normalized_score_diff, num_gradient_steps)

    return mean_loss, mean_acc, mean_top5_acc, mean_score_diff, mean_normalized_score_diff


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    if pad_sizes.numel() == 1:
        return input_.unsqueeze(0)
    max_pad_size = int(pad_sizes.max().item())
    output = input_.split(pad_sizes.detach().cpu().tolist())
    output = torch.stack(
        [
            F.pad(slice_, (0, max_pad_size - slice_.size(0)), "constant", pad_value)
            for slice_ in output
        ],
        dim=0,
    )
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the MILP branching policy.")
    parser.add_argument("--dataset", type=str, default="set_cover", help="Dataset key.")
    parser.add_argument("--cfg_idx", type=int, default=0, help="Configuration index.")
    parser.add_argument("--config_root", type=str, default="./cfg", help="Directory containing configuration files.")
    parser.add_argument("--model_suffix", type=str, default="", help="Optional suffix appended to model/log directories.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")
    parser.add_argument("--resume_model_dir", type=str, default="", help="Directory containing checkpoints to resume from.")
    parser.add_argument(
        "--eval_every",
        type=int,
        default=-1234,
        help="Evaluation frequency in gradient steps. Disabled if <= 0.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=140000,
        help="Checkpoint frequency in gradient steps. Disabled if <= 0.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=140000,
        help="Logging frequency in gradient steps. Disabled if <= 0.",
    )
    parser.add_argument(
        "--c_11",
        type=float,
        default=argparse.SUPPRESS,
        help="Tier-aware pairwise loss coefficient for tier1-tier1 pairs.",
    )
    parser.add_argument(
        "--c_12",
        type=float,
        default=argparse.SUPPRESS,
        help="Tier-aware pairwise loss coefficient for tier1-tier2 pairs.",
    )
    parser.add_argument(
        "--c_21",
        type=float,
        default=argparse.SUPPRESS,
        help="Tier-aware pairwise loss coefficient for tier2-tier1 pairs.",
    )
    parser.add_argument(
        "--c_22",
        type=float,
        default=argparse.SUPPRESS,
        help="Tier-aware pairwise loss coefficient for tier2-tier2 pairs.",
    )
    parser.add_argument(
        "--regression_target",
        type=str,
        default="score",
        help="Regression target for loss_option=regression: one of {dual, obj, score}.",
    )
    parser.add_argument(
        "--include_parent",
        type=int,
        default=argparse.SUPPRESS,
        choices=[0, 1],
        help="When regression_target=obj, add parent objective MSE to the training loss.",
    )
    parser.add_argument(
        "--dual_option",
        type=int,
        default=1,
        choices=[1, 2],
        help="Regression target variant key: top8_regression_targets_option{dual_option}.",
    )
    parser.add_argument(
        "--universal_cutoffbound",
        type=float,
        default=1e6,
        help="Universal cutoffbound used for dual options 2 and 4 in post-process.",
    )
    parser.add_argument(
        "--use_cutoff_minimum",
        type=int,
        default=argparse.SUPPRESS,
        choices=[0, 1],
        help="Whether to apply cutoff-based torch.minimum operations in HeuristicPolicy post-process.",
    )
    parser.add_argument(
        "--train_tb_every",
        type=int,
        default=argparse.SUPPRESS,
        help="TensorBoard write frequency in gradient steps for train metrics.",
    )
    parser.add_argument(
        "--grad_log_every",
        type=int,
        default=argparse.SUPPRESS,
        help="TensorBoard write frequency for gradient/parameter diagnostics.",
    )
    parser.add_argument(
        "--postprocess_cache_size",
        type=int,
        default=argparse.SUPPRESS,
        help="Max number of post-process samples cached in memory.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=argparse.SUPPRESS,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--persistent_workers",
        type=int,
        choices=[0, 1],
        default=argparse.SUPPRESS,
        help="Enable DataLoader persistent workers.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=argparse.SUPPRESS,
        help="DataLoader prefetch_factor (workers only).",
    )
    parser.add_argument(
        "--pin_memory",
        type=int,
        choices=[0, 1],
        default=argparse.SUPPRESS,
        help="Enable DataLoader pin_memory.",
    )
    return parser.parse_args(argv)


def _load_config(config_root: Path, dataset: str, cfg_idx: int) -> Dict[str, Any]:
    cfg_path = config_root / f"{dataset}_{cfg_idx}"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    return cfg


def _merge_args_with_config(init_args, cfg: Dict[str, Any]):
    args_dict = {**cfg, **vars(init_args)}
    args = argparse.Namespace(**args_dict)

    args.model_id = f"{args.dataset}_cfg{args.cfg_idx}"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not hasattr(args, "use_cutoff_minimum"):
        args.use_cutoff_minimum = True
    args.use_cutoff_minimum = bool(args.use_cutoff_minimum)
    if not hasattr(args, "include_parent"):
        args.include_parent = False
    args.include_parent = bool(args.include_parent)
    if not hasattr(args, "grad_log_every"):
        args.grad_log_every = int(getattr(args, "train_tb_every", 100))
    if hasattr(args, "persistent_workers"):
        args.persistent_workers = bool(args.persistent_workers)
    if hasattr(args, "pin_memory"):
        args.pin_memory = bool(args.pin_memory)
    return args


def main(argv=None):
    init_args = parse_args(argv)
    cfg = _load_config(Path(init_args.config_root), init_args.dataset, init_args.cfg_idx)
    args = _merge_args_with_config(init_args, cfg)

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    set_seed(args.seed)
    eval_disabled = int(getattr(args, "eval_every", 0)) <= 0
    if eval_disabled:
        args.max_val_samples = 0
        args.max_test_samples = 0
    data = load_data(args)
    train_loader = data.get("train")
    val_loader = None if eval_disabled else data.get("val")
    cons_nfeats, edge_nfeats, var_nfeats = _infer_feature_dimensions(train_loader)

    policy = load_model(args, cons_nfeats, edge_nfeats, var_nfeats)
    optimizer = get_optimizer(args, policy)

    train_postprocess_interface = None
    val_postprocess_interface = None
    if hasattr(policy, "post_process"):
        train_loader, train_postprocess_interface = _wrap_loader_for_postprocess(
            train_loader,
            shuffle=bool(getattr(args, "train_shuffle", True)),
            dual_option=int(getattr(args, "dual_option", 1)),
            universal_cutoffbound=float(getattr(args, "universal_cutoffbound", 1e6)),
            max_cache_size=int(getattr(args, "postprocess_cache_size", 128)),
        )
        val_loader, val_postprocess_interface = _wrap_loader_for_postprocess(
            val_loader,
            shuffle=bool(getattr(args, "val_shuffle", False)),
            dual_option=int(getattr(args, "dual_option", 1)),
            universal_cutoffbound=float(getattr(args, "universal_cutoffbound", 1e6)),
            max_cache_size=int(getattr(args, "postprocess_cache_size", 128)),
        )

    base_model_dir = Path(getattr(args, "model_dir", "./models"))
    if getattr(args, "model", None):
        base_model_dir = base_model_dir / args.model
    base_log_dir = Path(getattr(args, "log_dir", "./logs"))
    model_id = getattr(args, "model_id", None)
    if model_id:
        base_model_dir = base_model_dir / model_id
        base_log_dir = base_log_dir / model_id
    model_suffix = getattr(args, "model_suffix", "")
    if model_suffix:
        base_model_dir = Path(f"{base_model_dir}_{model_suffix}")
        base_log_dir = Path(f"{base_log_dir}_{model_suffix}")

    resume_model_dir_value = getattr(args, "resume_model_dir", "")
    resume_dir = Path(resume_model_dir_value) if resume_model_dir_value else None
    load_model_dir = resume_dir if (resume_dir and resume_dir.exists()) else base_model_dir
    start_step = 0
    if getattr(args, "resume", False):
        print("Resuming training...")
        start_step = load_checkpoint(policy, optimizer, step="max", save_dir=str(load_model_dir), device=args.device)
        resume_tag = load_model_dir.name
        base_model_dir = Path(f"{base_model_dir}_resume_from_{resume_tag}")
        base_log_dir = Path(f"{base_log_dir}_resume_from_{resume_tag}")
        policy = policy.to(args.device)

    base_model_dir.mkdir(parents=True, exist_ok=True)
    base_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model is saved to {base_model_dir}, logs are saved to {base_log_dir}.")
    
    train(
        args,
        policy,
        optimizer,
        train_loader,
        start_step=start_step,
        model_dir=base_model_dir,
        log_dir=base_log_dir,
        val_dataloader=val_loader,
        train_postprocess_interface=train_postprocess_interface,
        val_postprocess_interface=val_postprocess_interface,
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes.")
