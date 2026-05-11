import torch
import torch.nn.functional as F


class ObjTrainer:
    def __init__(self):
        self.best_acc = 0
        self.patience = 0

    def train(self, dataloader, model, optimizer, device):
        model.train()
        device = torch.device(device)

        train_losses = torch.tensor(0.0, device=device)
        num_lp_graphs = 0
        for i, data in enumerate(dataloader):
            data = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in data.items()
            }

            pred_obj = model(data)
            pred_obj = pred_obj.reshape(-1)
            true_obj = data["target_obj"].reshape(-1)
            loss = F.mse_loss(pred_obj, true_obj)

            train_losses += loss.detach() * data["num_lp_graphs"]
            num_lp_graphs += data["num_lp_graphs"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses / max(1, num_lp_graphs)

    @torch.no_grad()
    def eval(self, dataloader, model, device):
        model.eval()
        device = torch.device(device)

        num_milp_graphs = 0
        num_lp_graphs = 0
        losses = torch.tensor(0.0, device=device)
        accs = torch.tensor(0.0, device=device)
        top5_accs = torch.tensor(0.0, device=device)
        score_diffs = torch.tensor(0.0, device=device)
        normalized_score_diffs = torch.tensor(0.0, device=device)

        for i, data in enumerate(dataloader):
            data = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in data.items()
            }

            pred_obj = model(data)
            pred_obj = pred_obj.reshape(-1)
            true_obj = data["target_obj"].reshape(-1)
            loss = F.mse_loss(pred_obj, true_obj)
            
            losses += loss.detach() * data["num_lp_graphs"]
            num_milp_graphs += data["num_milp_graphs"]
            num_lp_graphs += data["num_lp_graphs"]

            parent_obj = data["parent_obj"].reshape(-1)
            true_score_lp = data["target_score"].reshape(-1)
            topk_per_milp = data["topk_per_milp"].reshape(-1)
            k_max = int(data["top_k_max"])
            bsz = int(data["num_milp_graphs"])

            lengths = (2 * topk_per_milp).to(torch.long)
            starts = torch.cumsum(lengths, dim=0) - lengths
            milp_ids = torch.repeat_interleave(torch.arange(bsz, device=device), lengths)  # [0,...0,1,...,1,...,bsz]
            local_lp_ids = torch.arange(lengths.sum(), device=device) - torch.repeat_interleave(
                    starts, lengths
                )  # [0,1,2,...,15,0,1,2,...15,...] (2 * K1 + 2 * K2 + ...)
            pair_ids = local_lp_ids // 2  # [0,0,1,1,2,2,...] (2 * K1 + 2 * K2 + ...)
            branch_dirs = local_lp_ids % 2  # 0=down, 1=up [0,1,0,1,...] (2 * K1 + 2 * K2 + ...)

            pred_down = torch.zeros((bsz, k_max), device=device, dtype=pred_obj.dtype)
            pred_up = torch.zeros((bsz, k_max), device=device, dtype=pred_obj.dtype)
            true_down = torch.zeros((bsz, k_max), device=device, dtype=true_score_lp.dtype)
            true_up = torch.zeros((bsz, k_max), device=device, dtype=true_score_lp.dtype)
            parent_down = torch.zeros((bsz, k_max), device=device, dtype=parent_obj.dtype)
            parent_up = torch.zeros((bsz, k_max), device=device, dtype=parent_obj.dtype)

            down_mask = branch_dirs == 0
            up_mask = ~down_mask

            pred_down[milp_ids[down_mask], pair_ids[down_mask]] = pred_obj[down_mask]
            pred_up[milp_ids[up_mask], pair_ids[up_mask]] = pred_obj[up_mask]
            true_down[milp_ids[down_mask], pair_ids[down_mask]] = true_score_lp[down_mask]
            true_up[milp_ids[up_mask], pair_ids[up_mask]] = true_score_lp[up_mask]
            parent_down[milp_ids[down_mask], pair_ids[down_mask]] = parent_obj[down_mask]
            parent_up[milp_ids[up_mask], pair_ids[up_mask]] = parent_obj[up_mask]

            parent = 0.5 * (parent_down + parent_up)
            gain_down = torch.clamp(pred_down - parent, min=1e-9)
            gain_up = torch.clamp(pred_up - parent, min=1e-9)
            pred_score = gain_down * gain_up
            true_score = 0.5 * (true_down + true_up)

            true_bestscore = true_score.max(dim=-1, keepdim=True).values
            pred_bestscore_index = pred_score.max(dim=-1, keepdim=True).indices
            pred_bestscore = true_score.gather(-1, pred_bestscore_index)
            pred_top5score = true_score.gather(
                    -1, pred_score.topk(min(5, pred_score.size(-1)), dim=-1).indices
                )

            accs += (pred_bestscore == true_bestscore).float().sum()
            top5_accs += (pred_top5score == true_bestscore).float().max(dim=-1).values.sum()
            score_diffs += (true_bestscore - pred_bestscore).abs().sum()
            normalized_score_diffs += (
                    (true_bestscore - pred_bestscore) / true_bestscore.clamp_min(1e-9)
                ).sum()

        return (
            losses / max(1, num_lp_graphs),
            accs / max(1, num_milp_graphs),
            top5_accs / max(1, num_milp_graphs),
            score_diffs / max(1, num_milp_graphs),
            normalized_score_diffs / max(1, num_milp_graphs),
        )


class RealObjTrainer:
    def __init__(self):
        self.best_acc = 0
        self.patience = 0

    def train(self, dataloader, model, optimizer, device):
        model.train()
        device = torch.device(device)

        train_losses = torch.tensor(0.0, device=device)
        num_lp_graphs = 0
        for i, data in enumerate(dataloader):
            data = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in data.items()
            }

            pred_obj = model(data)
            pred_obj = pred_obj.reshape(-1)
            cutoffbound = data["cutoffbound"].reshape(-1).to(device=pred_obj.device, dtype=pred_obj.dtype)
            # pred_obj = torch.minimum(pred_obj, cutoffbound)
            true_obj = data["target_obj"].reshape(-1)

            # if torch.any(true_obj > cutoffbound + 1e-6).item():
            #     print(f"true_obj: {true_obj} > cutoffbound {cutoffbound} + 1e-6")
            #     raise ValueError(f"true obj shouldn't be greater than cutoffbound. dataset incorrect somewhere")

            valid_mask = true_obj < cutoffbound + 1e-6
            valid_count = valid_mask.sum()
            global_valid_count = valid_count.detach().clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(global_valid_count, op=torch.distributed.ReduceOp.SUM)
            if int(global_valid_count.item()) == 0:
                continue
            if int(valid_count.item()) > 0:
                loss_sum = F.mse_loss(pred_obj[valid_mask], true_obj[valid_mask], reduction="sum")
                loss = loss_sum / valid_count
            else:
                loss_sum = pred_obj.sum() * 0.0
                loss = loss_sum

            train_losses += loss_sum.detach()
            num_lp_graphs += int(valid_count.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            num_lp_graphs_tensor = torch.as_tensor(float(num_lp_graphs), device=device)
            torch.distributed.all_reduce(train_losses, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(num_lp_graphs_tensor, op=torch.distributed.ReduceOp.SUM)
            num_lp_graphs = int(num_lp_graphs_tensor.item())

        return train_losses / max(1, num_lp_graphs)

    @torch.no_grad()
    def eval(self, dataloader, model, device):
        model.eval()
        device = torch.device(device)

        num_milp_graphs = 0
        num_lp_graphs = 0
        losses = torch.tensor(0.0, device=device)
        accs = torch.tensor(0.0, device=device)
        top5_accs = torch.tensor(0.0, device=device)
        score_diffs = torch.tensor(0.0, device=device)
        normalized_score_diffs = torch.tensor(0.0, device=device)

        for i, data in enumerate(dataloader):
            data = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in data.items()
            }

            pred_obj = model(data)
            pred_obj = pred_obj.reshape(-1)
            cutoffbound = data["cutoffbound"].reshape(-1).to(device=pred_obj.device, dtype=pred_obj.dtype)
            pred_obj = torch.minimum(pred_obj, cutoffbound)
            true_obj = data["target_obj"].reshape(-1)
            loss = F.mse_loss(pred_obj, true_obj)
            
            losses += loss.detach() * data["num_lp_graphs"]
            num_milp_graphs += data["num_milp_graphs"]
            num_lp_graphs += data["num_lp_graphs"]

            parent_obj = data["parent_obj"].reshape(-1)
            true_score_lp = data["target_score"].reshape(-1)
            topk_per_milp = data["topk_per_milp"].reshape(-1)
            k_max = int(data["top_k_max"])
            bsz = int(data["num_milp_graphs"])

            lengths = (2 * topk_per_milp).to(torch.long)
            starts = torch.cumsum(lengths, dim=0) - lengths
            milp_ids = torch.repeat_interleave(torch.arange(bsz, device=device), lengths)  # [0,...0,1,...,1,...,bsz]
            local_lp_ids = torch.arange(lengths.sum(), device=device) - torch.repeat_interleave(
                    starts, lengths
                )  # [0,1,2,...,15,0,1,2,...15,...] (2 * K1 + 2 * K2 + ...)
            pair_ids = local_lp_ids // 2  # [0,0,1,1,2,2,...] (2 * K1 + 2 * K2 + ...)
            branch_dirs = local_lp_ids % 2  # 0=down, 1=up [0,1,0,1,...] (2 * K1 + 2 * K2 + ...)

            pred_down = torch.zeros((bsz, k_max), device=device, dtype=pred_obj.dtype)
            pred_up = torch.zeros((bsz, k_max), device=device, dtype=pred_obj.dtype)
            true_down = torch.zeros((bsz, k_max), device=device, dtype=true_score_lp.dtype)
            true_up = torch.zeros((bsz, k_max), device=device, dtype=true_score_lp.dtype)
            parent_down = torch.zeros((bsz, k_max), device=device, dtype=parent_obj.dtype)
            parent_up = torch.zeros((bsz, k_max), device=device, dtype=parent_obj.dtype)

            down_mask = branch_dirs == 0
            up_mask = ~down_mask

            pred_down[milp_ids[down_mask], pair_ids[down_mask]] = pred_obj[down_mask]
            pred_up[milp_ids[up_mask], pair_ids[up_mask]] = pred_obj[up_mask]
            true_down[milp_ids[down_mask], pair_ids[down_mask]] = true_score_lp[down_mask]
            true_up[milp_ids[up_mask], pair_ids[up_mask]] = true_score_lp[up_mask]
            parent_down[milp_ids[down_mask], pair_ids[down_mask]] = parent_obj[down_mask]
            parent_up[milp_ids[up_mask], pair_ids[up_mask]] = parent_obj[up_mask]

            parent = 0.5 * (parent_down + parent_up)
            gain_down = torch.clamp(pred_down - parent, min=1e-9)
            gain_up = torch.clamp(pred_up - parent, min=1e-9)
            pred_score = gain_down * gain_up
            true_score = 0.5 * (true_down + true_up)

            true_bestscore = true_score.max(dim=-1, keepdim=True).values
            pred_bestscore_index = pred_score.max(dim=-1, keepdim=True).indices
            pred_bestscore = true_score.gather(-1, pred_bestscore_index)
            pred_top5score = true_score.gather(
                    -1, pred_score.topk(min(5, pred_score.size(-1)), dim=-1).indices
                )

            accs += (pred_bestscore == true_bestscore).float().sum()
            top5_accs += (pred_top5score == true_bestscore).float().max(dim=-1).values.sum()
            score_diffs += (true_bestscore - pred_bestscore).abs().sum()
            normalized_score_diffs += (
                    (true_bestscore - pred_bestscore) / true_bestscore.clamp_min(1e-9)
                ).sum()

        return (
            losses / max(1, num_lp_graphs),
            accs / max(1, num_milp_graphs),
            top5_accs / max(1, num_milp_graphs),
            score_diffs / max(1, num_milp_graphs),
            normalized_score_diffs / max(1, num_milp_graphs),
        )


class DeltaObjTrainer:
    def __init__(self):
        self.best_acc = 0
        self.patience = 0

    def train(self, dataloader, model, optimizer, device):
        model.train()
        device = torch.device(device)

        train_losses = torch.tensor(0.0, device=device)
        num_lp_graphs = 0
        for i, data in enumerate(dataloader):
            data = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in data.items()
            }

            pred_delta = model(data)
            pred_delta = pred_delta.reshape(-1)
            true_obj = data["target_obj"].reshape(-1)
            parent_obj = data["parent_obj"].reshape(-1)
            true_delta = true_obj - parent_obj
            loss = F.mse_loss(pred_delta, true_delta)

            train_losses += loss.detach() * data["num_lp_graphs"]
            num_lp_graphs += data["num_lp_graphs"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

        return train_losses / max(1, num_lp_graphs)

    @torch.no_grad()
    def eval(self, dataloader, model, device):
        model.eval()
        device = torch.device(device)

        num_milp_graphs = 0
        num_lp_graphs = 0
        losses = torch.tensor(0.0, device=device)
        accs = torch.tensor(0.0, device=device)
        top5_accs = torch.tensor(0.0, device=device)
        score_diffs = torch.tensor(0.0, device=device)
        normalized_score_diffs = torch.tensor(0.0, device=device)

        for i, data in enumerate(dataloader):
            data = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in data.items()
            }

            pred_delta = model(data)
            pred_delta = pred_delta.reshape(-1)
            true_obj = data["target_obj"].reshape(-1)
            parent_obj = data["parent_obj"].reshape(-1)
            true_delta = true_obj - parent_obj
            loss = F.mse_loss(pred_delta, true_delta)
            
            losses += loss.detach() * data["num_lp_graphs"]
            num_milp_graphs += data["num_milp_graphs"]
            num_lp_graphs += data["num_lp_graphs"]

            true_score_lp = data["target_score"].reshape(-1)
            topk_per_milp = data["topk_per_milp"].reshape(-1)
            k_max = int(data["top_k_max"])
            bsz = int(data["num_milp_graphs"])

            lengths = (2 * topk_per_milp).to(torch.long)
            starts = torch.cumsum(lengths, dim=0) - lengths
            milp_ids = torch.repeat_interleave(torch.arange(bsz, device=device), lengths)  # [0,...0,1,...,1,...,bsz]
            local_lp_ids = torch.arange(lengths.sum(), device=device) - torch.repeat_interleave(
                    starts, lengths
                )  # [0,1,2,...,15,0,1,2,...15,...] (2 * K1 + 2 * K2 + ...)
            pair_ids = local_lp_ids // 2  # [0,0,1,1,2,2,...] (2 * K1 + 2 * K2 + ...)
            branch_dirs = local_lp_ids % 2  # 0=down, 1=up [0,1,0,1,...] (2 * K1 + 2 * K2 + ...)

            pred_down = torch.zeros((bsz, k_max), device=device, dtype=pred_delta.dtype)
            pred_up = torch.zeros((bsz, k_max), device=device, dtype=pred_delta.dtype)
            true_down = torch.zeros((bsz, k_max), device=device, dtype=true_score_lp.dtype)
            true_up = torch.zeros((bsz, k_max), device=device, dtype=true_score_lp.dtype)

            down_mask = branch_dirs == 0
            up_mask = ~down_mask

            pred_down[milp_ids[down_mask], pair_ids[down_mask]] = pred_delta[down_mask]
            pred_up[milp_ids[up_mask], pair_ids[up_mask]] = pred_delta[up_mask]
            true_down[milp_ids[down_mask], pair_ids[down_mask]] = true_score_lp[down_mask]
            true_up[milp_ids[up_mask], pair_ids[up_mask]] = true_score_lp[up_mask]

            gain_down = torch.clamp(pred_down, min=1e-9)
            gain_up = torch.clamp(pred_up, min=1e-9)
            pred_score = gain_down * gain_up
            true_score = 0.5 * (true_down + true_up)

            true_bestscore = true_score.max(dim=-1, keepdim=True).values
            pred_bestscore_index = pred_score.max(dim=-1, keepdim=True).indices
            pred_bestscore = true_score.gather(-1, pred_bestscore_index)
            pred_top5score = true_score.gather(
                    -1, pred_score.topk(min(5, pred_score.size(-1)), dim=-1).indices
                )

            accs += (pred_bestscore == true_bestscore).float().sum()
            top5_accs += (pred_top5score == true_bestscore).float().max(dim=-1).values.sum()
            score_diffs += (true_bestscore - pred_bestscore).abs().sum()
            normalized_score_diffs += (
                    (true_bestscore - pred_bestscore) / true_bestscore.clamp_min(1e-9)
                ).sum()

        return (
            losses / max(1, num_lp_graphs),
            accs / max(1, num_milp_graphs),
            top5_accs / max(1, num_milp_graphs),
            score_diffs / max(1, num_milp_graphs),
            normalized_score_diffs / max(1, num_milp_graphs),
        )
