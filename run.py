import copy
import os
import subprocess

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn_data.collate_func import collate_fn_lp_base, collate_fn_lp_flat
from gnn_data.dataset import LPGraphDataset, LPDataset
from gnn_models import get_model
from trainer import DeltaObjTrainer, ObjTrainer, RealDeltaObjTrainer, RealObjTrainer
from utils.experiment import save_run_config, setup_wandb, count_parameters

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path='./config', config_name="lp")
def main(args: DictConfig):
    log_folder_name = save_run_config(args)
    setup_wandb(args)

    target = str(args.gnn.target).lower()
    use_real_obj = target in {"realobj", "real_obj"}
    use_real_delta_obj = target in {"realdeltaobj", "real_deltaobj", "real_delta_obj"}
    use_flat_train = bool(args.train.shuffle_lp) or use_real_obj or use_real_delta_obj
    train_set = (
        LPGraphDataset(args.train.datapath, 'train', transform=None)
        if use_flat_train
        else LPDataset(args.train.datapath, 'train', transform=None)
    )
    valid_set = LPDataset(args.train.datapath, 'valid', transform=None)
    test_set = LPDataset(args.train.datapath, 'test', transform=None)

    if args.train.debug:
        train_set = train_set[:10000]
        valid_set = valid_set[:1000]
        test_set = test_set[:1000]

    train_loader = DataLoader(train_set,
                      batch_size=args.train.batchsize,
                      shuffle=True,
                      collate_fn=collate_fn_lp_flat if use_flat_train else collate_fn_lp_base,
                      num_workers=8, persistent_workers=1, prefetch_factor=2,
                      pin_memory=True)
    eval_batchsize = max(1, args.train.batchsize // 16) if use_flat_train else args.train.batchsize
    val_loader = DataLoader(valid_set,
                            batch_size=eval_batchsize,
                            shuffle=False,
                            collate_fn=collate_fn_lp_base,
                            num_workers=8, persistent_workers=1, prefetch_factor=2,
                            pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=eval_batchsize,
                             shuffle=False,
                             collate_fn=collate_fn_lp_base,
                             num_workers=8, persistent_workers=1, prefetch_factor=2,
                             pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_val_accs = []
    test_losses = []
    test_accs = []
    test_top5_accs = []
    test_score_diffs = []
    test_normalized_score_diffs = []

    for run in range(args.train.runs):
        model = get_model(args.gnn).to(device)
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='max', 
                                                         factor=0.5,
                                                         patience=int(args.train.patience * 0.6),
                                                         min_lr=1.e-5)
        if target == 'obj':
            trainer = ObjTrainer()
        elif use_real_obj:
            trainer = RealObjTrainer()
        elif target == 'deltaobj':
            trainer = DeltaObjTrainer()
        elif use_real_delta_obj:
            trainer = RealDeltaObjTrainer()
        else:
            raise ValueError(f"Unsupported gnn.target: {args.gnn.target}")

        print_every = 5
        pbar = tqdm(range(args.train.epoch), miniters=print_every)
        for epoch in pbar:
            train_loss = trainer.train(train_loader, model, optimizer, device).item()
            val_loss, val_acc, val_top5_acc, val_score_diff, val_normalized_score_diff = trainer.eval(val_loader, model, device)
            val_loss = val_loss.item()
            val_acc = val_acc.item()
            val_top5_acc = val_top5_acc.item()
            val_score_diff = val_score_diff.item()
            val_normalized_score_diff = val_normalized_score_diff.item()

            if scheduler is not None:
                scheduler.step(val_acc)
            
            if trainer.best_acc < val_acc:
                trainer.patience = 0
                trainer.best_acc = val_acc
                best_model = copy.deepcopy(model.state_dict())
                if args.train.ckpt:
                    torch.save(model.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
            else:
                trainer.patience += 1

            if trainer.patience > args.train.patience:
                break

            stats_dict = {'train_loss': train_loss,
                          'val_loss': val_loss,
                          'val_acc': val_acc,
                          'val_top5_acc': val_top5_acc,
                          'val_score_diff': val_score_diff, 
                          'val_normalized_score_diff': val_normalized_score_diff, 
                          'lr': scheduler.optimizer.param_groups[0]["lr"]}

            should_print = (
                epoch == 0
                or (epoch + 1) % print_every == 0
                or epoch + 1 == args.train.epoch
            )
            pbar.set_postfix(stats_dict, refresh=should_print)
            wandb.log(stats_dict)

        model.load_state_dict(best_model)
        test_loss, test_acc, test_top5_acc, test_score_diff, test_normalized_score_diff = trainer.eval(test_loader, model, device)

        best_val_accs.append(trainer.best_acc)
        test_losses.append(test_loss.item())
        test_accs.append(test_acc.item())
        test_top5_accs.append(test_top5_acc.item())
        test_score_diffs.append(test_score_diff.item())
        test_normalized_score_diffs.append(test_normalized_score_diff.item())

    wandb.log({
        'num_params': count_parameters(model),
        'best_val_accs': np.mean(best_val_accs),
        'test_loss_mean': np.mean(test_losses),
        'test_loss_std': np.std(test_losses),

        'test_acc_mean': np.mean(test_accs),
        'test_acc_std': np.std(test_accs),

        'test_top5_acc_mean': np.mean(test_top5_accs),
        'test_top5_acc_std': np.std(test_top5_accs),

        'test_score_diff_mean': np.mean(test_score_diffs),
        'test_score_diff_std': np.std(test_score_diffs),

        'test_normalized_score_diff_mean': np.mean(test_normalized_score_diffs),
        'test_normalized_score_diff_std': np.std(test_normalized_score_diffs),
    })


if __name__ == '__main__':
    main()
