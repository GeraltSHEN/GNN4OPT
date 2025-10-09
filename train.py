import os
import pickle
import time
from pathlib import Path
from typing import Dict, Optional

import psutil
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from utils import get_optimizer, load_model


def log_cpu_memory_usage(epoch: int, step: Optional[str] = None):
    """Report CPU memory usage at coarse intervals."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    if epoch == 1 or epoch % 1 == 0:
        tag = f", Step {step}" if step is not None else ""
        print(
            f"[Epoch {epoch}{tag}] CPU Memory - RSS: {memory_info.rss / (1024 ** 3):.2f} GB"
        )


def _to_data_list(batch) -> list[HeteroData]:
    if hasattr(batch, "to_data_list"):
        return batch.to_data_list()
    return [batch]


def _forward_sample(
    model: torch.nn.Module,
    sample: HeteroData,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample = sample.to(device)
    candidate_indices = sample.candidate_indices.to(device)
    logits = model(sample, candidate_indices)
    target = sample.candidate_choice.to(device)
    return logits, target


def train_one_epoch(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    if loader is None:
        return {"loss": 0.0, "accuracy": 0.0}

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        samples = _to_data_list(batch)
        if not samples:
            continue

        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        batch_correct = 0

        for sample in samples:
            logits, target = _forward_sample(model, sample, device)
            sample_loss = F.cross_entropy(
                logits.unsqueeze(0),
                target.view(1),
            )
            batch_loss = batch_loss + sample_loss

            pred = logits.argmax(dim=-1)
            batch_correct += int((pred == target).item())

        batch_size = len(samples)
        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * batch_size
        total_correct += batch_correct
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return {"loss": avg_loss, "accuracy": accuracy}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
) -> Dict[str, float]:
    if loader is None:
        return {"loss": 0.0, "accuracy": 0.0}

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        samples = _to_data_list(batch)
        for sample in samples:
            logits, target = _forward_sample(model, sample, device)
            sample_loss = F.cross_entropy(
                logits.unsqueeze(0),
                target.view(1),
            )
            total_loss += sample_loss.item()
            pred = logits.argmax(dim=-1)
            total_correct += int((pred == target).item())
        total_samples += len(samples)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return {"loss": avg_loss, "accuracy": accuracy}


def run_training(args, data: Dict[str, Optional[DataLoader]]):
    device = torch.device(getattr(args, "device", "cpu"))
    model = load_model(args)
    optimizer = get_optimizer(args, model)

    train_loader = data.get("train")
    val_loader = data.get("val")
    test_loader = data.get("test")

    num_epochs = getattr(args, "n_epochs", getattr(args, "epochs", 1))
    best_val_acc = float("-inf")
    best_state = None
    metrics_history = []
    print_freq = getattr(args, "resultPrintFreq", 1)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        elapsed = time.time() - start_time

        val_metrics = evaluate(model, val_loader, device) if val_loader else None
        if val_metrics:
            val_acc = val_metrics["accuracy"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        metrics_history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"] if val_metrics else None,
                "val_acc": val_metrics["accuracy"] if val_metrics else None,
                "time": elapsed,
            }
        )

        if epoch % max(1, print_freq) == 0 or epoch == num_epochs:
            log = [
                f"Epoch {epoch:03d}/{num_epochs}",
                f"train_loss={train_metrics['loss']:.4f}",
                f"train_acc={train_metrics['accuracy']:.4f}",
                f"time={elapsed:.2f}s",
            ]
            if val_metrics:
                log.extend(
                    [
                        f"val_loss={val_metrics['loss']:.4f}",
                        f"val_acc={val_metrics['accuracy']:.4f}",
                    ]
                )
            print(" | ".join(log))

        log_cpu_memory_usage(epoch, step="training")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device) if test_loader else None
    if test_metrics:
        print(
            f"Test metrics -> loss: {test_metrics['loss']:.4f}, "
            f"accuracy: {test_metrics['accuracy']:.4f}"
        )

    model_id = getattr(args, "model_id", None)
    if metrics_history and model_id:
        log_dir = Path("./logs/run_training")
        log_dir.mkdir(parents=True, exist_ok=True)
        history_path = log_dir / f"{model_id}_TrainingStats.pkl"
        with history_path.open("wb") as f:
            pickle.dump(metrics_history, f)
        if test_metrics:
            test_path = log_dir / f"{model_id}_TestStats.pkl"
            with test_path.open("wb") as f:
                pickle.dump(test_metrics, f)

    return {
        "model": model,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
