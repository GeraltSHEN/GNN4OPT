import copy
import os
import subprocess

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gnn_data.collate_func import collate_fn_lp_base
from gnn_data.dataset import LPDataset
from gnn_models import get_model
from trainer import DeltaObjTrainer, ObjTrainer
from utils.experiment import save_run_config, setup_wandb, count_parameters

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path='./config', config_name="lp")
def main(args: DictConfig):
    world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
    rank = int(os.environ['RANK'])  # Rank of the current process
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size > 1, "This running file for multi gpu usage only!!!!"

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=local_rank)

    train_set = LPDataset(args.train.datapath, 'train', transform=None)
    valid_set = LPDataset(args.train.datapath, 'valid', transform=None)
    test_set = LPDataset(args.train.datapath, 'test', transform=None)

    if args.train.debug:
        train_set = train_set[:10000]
        valid_set = valid_set[:1000]
        test_set = test_set[:1000]
    
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(valid_set, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_set,
                            batch_size=args.train.batchsize // world_size,
                            collate_fn=collate_fn_lp_base,
                            num_workers=8, persistent_workers=1, prefetch_factor=2,
                            pin_memory=True,
                            sampler=train_sampler)
    val_loader = DataLoader(valid_set,
                            batch_size=args.train.batchsize // world_size,
                            collate_fn=collate_fn_lp_base,
                            num_workers=8, persistent_workers=1, prefetch_factor=2,
                            pin_memory=True,
                            sampler=val_sampler)
    test_loader = DataLoader(test_set,
                             batch_size=args.train.batchsize // world_size,
                             collate_fn=collate_fn_lp_base,
                             num_workers=8, persistent_workers=1, prefetch_factor=2,
                             pin_memory=True,
                             sampler=test_sampler)
    if rank == 0:
        log_folder_name = save_run_config(args)
        setup_wandb(args)
        best_val_accs = []
        test_losses = []
        test_accs = []
        test_top5_accs = []
        test_score_diffs = []
        test_normalized_score_diffs = []

    for run in range(args.train.runs):
        torch.cuda.empty_cache()
        dist.barrier()

        model = get_model(args.gnn).to(local_rank)
        model = DistributedDataParallel(model, device_ids=[local_rank])
        best_model = copy.deepcopy(model.state_dict())

        optimizer = optim.Adam(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='max',
                                                         factor=0.5,
                                                         patience=int(args.train.patience * 0.6),
                                                         min_lr=1.e-5)
        if args.gnn.target == 'obj':
            trainer = ObjTrainer()
        elif args.gnn.target == 'deltaobj':
            trainer = DeltaObjTrainer()
        else:
            raise ValueError(f"Unsupported gnn.target: {args.gnn.target}")

        for epoch in range(args.train.epoch):
            train_sampler.set_epoch(epoch)
            train_loss = trainer.train(train_loader, model, optimizer, local_rank)
            val_loss, val_acc, val_top5_acc, val_score_diff, val_normalized_score_diff = trainer.eval(val_loader, model, local_rank)

            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_acc, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_top5_acc, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_score_diff, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_normalized_score_diff, op=dist.ReduceOp.AVG)

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
                if args.train.ckpt and rank == 0:
                    torch.save(model.module.state_dict(), os.path.join(log_folder_name, f'best_model{run}.pt'))
            else:
                trainer.patience += 1

            if trainer.patience > args.train.patience:
                break

            if rank == 0:
                stats_dict = {'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'val_top5_acc': val_top5_acc,
                        'val_score_diff': val_score_diff, 
                        'val_normalized_score_diff': val_normalized_score_diff, 
                        'lr': scheduler.optimizer.param_groups[0]["lr"]}
                wandb.log(stats_dict)
                logger.info(', '.join([k + f':{v:.5f}' for k, v in stats_dict.items()]))

        dist.barrier()
        model.load_state_dict(best_model)
        test_loss, test_acc, test_top5_acc, test_score_diff, test_normalized_score_diff = trainer.eval(test_loader, model, local_rank)
        dist.all_reduce(test_loss, op=dist.ReduceOp.AVG)
        dist.all_reduce(test_acc, op=dist.ReduceOp.AVG)
        dist.all_reduce(test_top5_acc, op=dist.ReduceOp.AVG)
        dist.all_reduce(test_score_diff, op=dist.ReduceOp.AVG)
        dist.all_reduce(test_normalized_score_diff, op=dist.ReduceOp.AVG)
        dist.barrier()
        test_acc = test_acc.item()
        test_loss = test_loss.item()

        if rank == 0:
            best_val_accs.append(trainer.best_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_top5_accs.append(test_top5_acc.item())
            test_score_diffs.append(test_score_diff.item())
            test_normalized_score_diffs.append(test_normalized_score_diff.item())

    if rank == 0:
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

    dist.barrier()
    # at the very end
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
