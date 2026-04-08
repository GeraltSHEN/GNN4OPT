import torch
import torch.nn.functional as F


class DualTrainer:
    def __init__(self):
        self.best_acc = 0
        self.patience = 0

    def train(self, dataloader, model, optimizer, device):
        model.train()
        device = torch.device(device)

        train_losses = 0.
        num_lp_graphs = 0
        for i, data in enumerate(dataloader):
            data = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in data.items()
            }

            pred_obj = model(data)
            if isinstance(pred_obj, dict):
                pred_obj = pred_obj["pred_obj"]
            true_obj = data["target_obj"]
            loss = F.mse_loss(pred_obj, true_obj)

            train_losses += loss.detach() * data["num_lp_graphs"]
            num_lp_graphs += data["num_lp_graphs"]

            optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses / max(1, num_lp_graphs)

    @torch.no_grad()
    def eval(self, dataloader, model, device):
        model.eval()
        device = torch.device(device)

        num_milp_graphs = 0
        num_lp_graphs = 0
        losses = 0.
        accs = 0.
        top5_accs = 0.
        score_diffs = 0.
        normalized_score_diffs = 0.

        for i, data in enumerate(dataloader):
            data = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in data.items()
            }

            pred_obj = model(data)
            if isinstance(pred_obj, dict):
                pred_obj = pred_obj["pred_obj"]
            true_obj = data["target_obj"]
            true_parent_obj = data["parent_obj"]
            first_half = data["num_milp_graphs"]
            gain_down = torch.clamp(pred_obj[:first_half, :] - true_parent_obj.unsqueeze(1), min=1e-9)
            gain_up = torch.clamp(pred_obj[first_half:, :] - true_parent_obj.unsqueeze(1), min=1e-9)
            pred_score = gain_down * gain_up
        
            true_score = data["target_score"]
            true_bestscore = true_score.max(dim=-1, keepdims=True).values
            # use the rank to compute acc. instead of value of pred_score
            pred_bestscore_index = pred_score.max(dim=-1, keepdims=True).indices
            pred_bestscore = true_score.gather(-1, pred_bestscore_index)

            pred_top5score_index = pred_score.topk(min(5, pred_score.size(-1))).indices
            pred_top5score = true_score.gather(-1, pred_top5score_index)

            loss = F.mse_loss(pred_obj, true_obj)
            acc = (pred_bestscore == true_bestscore).float()
            top5_acc = (pred_top5score == true_bestscore).float().max(dim=-1).values
            score_diff = (true_bestscore - pred_bestscore).abs()
            normalized_score_diff = ((true_bestscore - pred_bestscore) / true_bestscore.clamp_min(1e-9))

            num_milp_graphs += data["num_milp_graphs"]
            num_lp_graphs += data["num_lp_graphs"]

            losses += loss * data["num_lp_graphs"]
            accs += acc.sum()
            top5_accs += top5_acc.sum()
            score_diffs += score_diff.sum()
            normalized_score_diffs += normalized_score_diff.sum() 

        return (
            losses / max(1, num_lp_graphs),
            accs / max(1, num_milp_graphs),
            top5_accs / max(1, num_milp_graphs),
            score_diffs / max(1, num_milp_graphs),
            normalized_score_diffs / max(1, num_milp_graphs),
        )
