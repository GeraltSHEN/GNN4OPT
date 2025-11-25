import time
import argparse
import os
from pathlib import Path
import pdb
from typing import Any, Dict, Optional

import psutil
import torch
import torch.nn.functional as F
import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils import (
    get_optimizer,
    load_model,
    load_data,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    print_dash_str
)


def log_cpu_memory_usage(epoch: int, step: Optional[str] = None):
    """Report CPU memory usage at coarse intervals."""
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
):
    policy.train()
    device = args.device
    epochs = args.epochs
    eval_every = args.eval_every
    save_every = args.save_every
    print_every = args.print_every
    loss_option = args.loss_option
    score_th = float('inf')

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
                    num_gradient_steps
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

            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                candidates=batch.candidates,
                n_constraints_per_graph=batch.n_constraints_per_graph,
                n_variables_per_graph=batch.n_variables_per_graph,
            )

            if score_th < float("inf"):
                select_indices = (
                    batch.candidate_scores.max(axis=-1).values < score_th
                )
                logits = logits[select_indices]
                batch = batch[select_indices]
                if len(logits) == 0:
                    continue
            else:
                # Index the results by the candidates, and split and pad them
                logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)

            if loss_option == "classification":
                loss = F.cross_entropy(logits, batch.candidate_choices)
            elif loss_option == "regression":
                loss = F.mse_loss(logits, batch.candidate_scores)
            else:
                raise ValueError(f"Unsupported loss option: {loss_option}")

            optimizer.zero_grad()
            loss.backward()
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
            writer.add_scalar("Loss/train", loss.item(), num_gradient_steps)
            writer.add_scalar("Accuracy/train", accuracy, num_gradient_steps)
            writer.add_scalar("Top5_Accuracy/train", top5_acc, num_gradient_steps)
            # New stats
            score_diff = (true_bestscore - true_scores.gather(-1, predicted_bestindex).clip(0)).mean().item()
            normalized_score_diff = ((true_bestscore - true_scores.gather(-1, predicted_bestindex).clip(0)) / true_bestscore).mean().item()
            mean_score_diff += score_diff * batch.num_graphs
            mean_normalized_score_diff += normalized_score_diff * batch.num_graphs

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


def evaluate(policy, data_loader, device, writer, num_gradient_steps):
    mean_loss = 0
    mean_acc = 0
    mean_top5_acc = 0
    mean_score_diff = 0
    mean_normalized_score_diff = 0

    policy.eval()

    n_samples_processed = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, disable=True):
            batch = batch.to(device)
            logits = policy(
                batch.constraint_features,
                batch.edge_index,
                batch.edge_attr,
                batch.variable_features,
                candidates=batch.candidates,
                n_constraints_per_graph=batch.n_constraints_per_graph,
                n_variables_per_graph=batch.n_variables_per_graph,
            )
            # Index the results by the candidates, and split and pad them
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
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
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
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
        default=1000,
        help="Evaluation frequency in gradient steps. Disabled if <= 0.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10000,
        help="Checkpoint frequency in gradient steps. Disabled if <= 0.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=1000,
        help="Logging frequency in gradient steps. Disabled if <= 0.",
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
    return args


def main(argv=None):
    init_args = parse_args(argv)
    cfg = _load_config(Path(init_args.config_root), init_args.dataset, init_args.cfg_idx)
    args = _merge_args_with_config(init_args, cfg)

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    data = load_data(args)
    set_seed(args.seed)
    train_loader = data.get("train")
    val_loader = data.get("val")
    cons_nfeats, edge_nfeats, var_nfeats = _infer_feature_dimensions(train_loader)

    policy = load_model(args, cons_nfeats, edge_nfeats, var_nfeats)
    optimizer = get_optimizer(args, policy)

    base_model_dir = Path(getattr(args, "model_dir", "./models"))
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
    )


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes.")
