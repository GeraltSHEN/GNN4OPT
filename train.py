
from typing import Callable
import torch
import torch.nn.functional as F
import csv
import numpy as np
from torch_geometric.utils import (negative_sampling)

import time
import os
import pickle
import matplotlib.pyplot as plt

import psutil

from utils import *


def log_cpu_memory_usage(epoch, step=None):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    if epoch % 50 == 0:
        print(f"[Epoch {epoch}{f', Step {step}' if step is not None else ''}] CPU Memory - RSS: {memory_info.rss / (1024 ** 3):.2f} GB")


def run_training(args, data):
    model = load_model(args)
    if args.continue_training:
        print('loading pretrained model weights')
        model = load_weights(model, args)
    print(f'----- {args.model_id} in {args.dataset} dataset -----')
    print('#params:', sum(p.numel() for p in model.parameters()))
    optimizer = get_optimizer(args, model)
    print('loss_type: ', args.loss_type)

    start_time = time.time()
    ##############################################################################################################
    Learning(args, data, model, optimizer)
    ##############################################################################################################
    end_time = time.time()
    training_time = end_time - start_time
    print(f'----time required for {args.epochs} epochs training: {round(training_time)}s----')
    print(f'----time required for {args.epochs} epochs training: {round(training_time / 60)}min----')
    print(f'----time required for {args.epochs} epochs training: {round(training_time / 3600)}hr----')
    # check the model on the test set
    scores = evaluate_model(args, data)


def train_model(optimizer, model, args, data, problem, epoch_stats):
    for batch in (data['train'] if args.data_generator else data['train']):
        optimizer_step(model, optimizer, batch, args, problem, epoch_stats)


def optimizer_step(model, optimizer, batch, args, problem, epoch_stats):
    start_time = time.time()
    optimizer.zero_grad()
    train_loss = get_loss(model, batch, problem, args, args.loss_type)
    train_loss.backward()
    optimizer.step()
    train_time = time.time() - start_time
    dict_agg(epoch_stats, 'train_time', train_time)
    dict_agg(epoch_stats, 'train_loss', float(train_loss.detach().cpu()))
    dict_agg(epoch_stats, 'train_agg', 1.)


def get_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sample_negatives: bool = False,
):
    def get_negatives(train_data, pos_y_coos, pos_y_values):
        neg_y_coos = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=pos_y_coos.size(-1),
            method="sparse",
        )

        y_coos = torch.cat(
            [pos_y_coos, neg_y_coos],
            dim=-1,
        )
        y_values = torch.cat([pos_y_values, torch.zeros_like(pos_y_values)], dim=0)
        return y_coos, y_values

    def fun(train_data, adj, y_dl) -> float:
        model.train()
        optimizer.zero_grad()
        tot_loss = 0
        for y_coos, y_values in y_dl:
            if sample_negatives:
                y_coos, y_values = get_negatives(train_data, y_coos, y_values)

            out = model(train_data, y_coos, adj)
            loss = loss_fn(out, y_values)
            loss.backward()
            tot_loss += float(loss)
        optimizer.step()
        return tot_loss

    return fun


def Learning(args, data, model, optimizer):
    best = float('inf')
    stats = {}

    # TODO:
    loss_fn = get_loss_fn(args)

    train_step = get_train_step(
        model, optimizer, loss_fn, sample_negatives=args.sample_negatives
    )

    train_y_dl = CooSampler(
        train_y.indices(), train_y.values(), batch_size=cfg.batch_size, shuffle=True
    )
    val_y_dl = CooSampler(
        val_y.indices(), val_y.values(), batch_size=val_y._nnz(), shuffle=False
    )

    for epoch in range(args.epochs):
        if args.data_generator and epoch % args.renew_freq == 0:
            data['train'].dataset.refresh()
        epoch_stats = {}
        # train
        model.train()
        start_time = time.time()
        train_loss = train_step(train_data, train_adj, train_y_dl)
        epoch_stats['train_loss'] = train_loss / len(train_y_dl)
        epoch_stats['train_time'] = time.time() - start_time
        curr_loss = epoch_stats['train_loss'] / epoch_stats['train_agg']
        log_cpu_memory_usage(epoch, 'training')
        # validate
        model.eval()
        start_time = time.time()
        val_metric = test(val_data, val_adj, val_y_dl)
        epoch_stats['val_metric'] = val_metric
        epoch_stats['val_time'] = time.time() - start_time
        # model checkpointg'])
        if val_metric < best:
            torch.save({'state_dict': model.state_dict()}, './models/' + args.model_id + '.pth')
            print(f'checkpoint saved at epoch {epoch}')
            best = val_metric

        # print epoch_stats
        if epoch % args.resultPrintFreq == 0 or epoch == args.epochs - 1:
            print('----- Epoch {} -----'.format(epoch))
            print('Train Loss: {:.5f}, '
                  'Train Time: {: .5f}'.format(epoch_stats['train_loss'],
                                               epoch_stats['train_time']))
            print('Val Metric: {:.5f}, '
                  'Val Time: {: .5f}'.format(epoch_stats['val_metric'],
                                             epoch_stats['val_time']))

        if epoch % args.resultSaveFreq == 0 or epoch == args.epochs - 1:
            stats = epoch_stats
            with open(f'./logs/run_training/{args.model_id}_TrainingStats.dict', 'wb') as f:
                pickle.dump(stats, f)


def featurize_batch(args, batch):
    with torch.no_grad():
        batch = batch.to(args.device)
        return batch


def get_loss_fn(args):
    # TODO
    return torch.nn.MSELoss()


def get_obj_loss(V, batch, problem, is_loss_batched):
    if problem.name == "primal":
        predicted_obj = problem.obj_fn(V=V,
                                       edge_index_triu=batch.edge_index_directed,
                                       num_edges=batch.n_edges,
                                       batched_obj=is_loss_batched)
        obj_loss = - predicted_obj / batch.num_graphs  # obj_fn gave positive objective value and negative loss is needed for training (minimization problem)
    elif problem.name == "dual":
        predicted_obj = problem.new_obj_fn(V=V, L=batch.L,
                                           batched_obj=is_loss_batched)
        obj_loss = predicted_obj / batch.num_graphs
    else:
        raise ValueError('Invalid problem')
    return obj_loss


def get_soln_loss(V, batch):
    # TODO: need validation
    bsz = batch.num_graphs
    V = V.view(bsz, batch.num_nodes, -1)
    predicted_soln = torch.bmm(V, V.transpose(1, 2))
    soln_loss = F.mse_loss(predicted_soln, batch.soln.view(bsz, batch.num_nodes, -1))
    return soln_loss


def get_gap(model, batch, problem, args, is_loss_batched):
    x, batch = featurize_batch(args, batch)
    V, p = model(x, batch.edge_index)
    if problem.name == "primal":
        predicted_obj = problem.obj_fn(V=V,
                                       edge_index_triu=batch.edge_index_directed,
                                       num_edges=batch.n_edges,
                                       batched_obj=is_loss_batched)
    elif problem.name == "dual":
        predicted_obj = problem.new_obj_fn(V=V, L=batch.L,
                                           batched_obj=is_loss_batched)
    else:
        raise ValueError('Invalid problem')
    gap = problem.optimality_gap(predicted_obj, batch.targets)
    return gap


def dict_agg(stats, key, value):
    if key in stats.keys():
        if "worst" in key:
            stats[key] = max(stats[key], value)
        else:
            stats[key] += value
    else:
        stats[key] = value


def evaluate_model(args, data, problem):
    test_stats = {}
    test_gaps = []
    model = load_model(args)
    model = load_weights(model, args)
    model.eval()
    for i, batch in enumerate(data['test']):
        start_time = time.time()
        gap = get_gap(model, batch, problem, args, is_loss_batched=True)
        test_time = time.time() - start_time

        gap_mean = float(gap.mean().detach().cpu())
        gap_worst = float(torch.norm(gap, float('inf')).detach().cpu())  # max of samples

        dict_agg(test_stats, 'test_time', test_time)
        dict_agg(test_stats, 'test_gap', gap_mean)
        dict_agg(test_stats, 'test_gap_worst', gap_worst)
        dict_agg(test_stats, 'test_agg', 1.)
        test_gaps.append(gap_mean)
    test_stats['test_gaps'] = np.array(test_gaps)

    with open(f'./logs/run_training/{args.model_id}_TestStats.dict', 'wb') as f:
        pickle.dump(test_stats, f)

    calculate_scores(args, data)


def calculate_scores(args, data):
    if os.path.exists(f'./logs/run_training/{args.model_id}_TrainingStats.dict'):
        try:
            with open(f'./logs/run_training/{args.model_id}_TrainingStats.dict', 'rb') as f:
                training_stats = pickle.load(f)
        except:
            print(f'{args.model_id}_TrainingStats.dict is missing. Load test stats only.')

    with open(f'./logs/run_training/{args.model_id}_TestStats.dict', 'rb') as f:
        test_stats = pickle.load(f)

    # store the test gap
    np.save(f'./data/results_summary/{args.model_id}_test_gaps.npy', test_stats['test_gaps'])

    scores = {
              'test_optimality_gap_mean': test_stats['test_gap'] / test_stats['test_agg'],
              'test_optimality_gap_worst': test_stats['test_gap_worst'],
              'train_time': training_stats['train_time'],
              'val_time': training_stats['val_time'] / training_stats['val_agg'],
              'test_time': test_stats['test_time'] / test_stats['test_agg'],}
    print(scores)
    create_report(scores, args)


def create_report(scores, args):
    args_dict = args_to_dict(args)
    # combine scores and args dict
    args_scores_dict = args_dict | scores
    # save dict
    save_dict(args_scores_dict, args)
    plot_distribution(args)


def args_to_dict(args):
    return vars(args)


def save_dict(dictionary, args):
    w = csv.writer(open('./data/results_summary/' + args.model_id + '.csv', 'w'))
    for key, val in dictionary.items():
        w.writerow([key, val])


def plot_distribution(args):
    """
    Plot the distribution of test gaps.
    """
    test_gaps = np.load(f'./data/results_summary/{args.model_id}_test_gaps.npy')
    plt.figure()
    plt.hist(test_gaps, bins=100)
    plt.xlabel('test gap')
    plt.savefig(f'./data/results_summary/{args.model_id}_test_gap_hist.pdf', format='pdf')


def load_weights(model, args):
    PATH = './models/' + args.model_id + '.pth'
    #checkpoint = torch.load(PATH, map_location=args.device, weights_only=False)
    checkpoint = torch.load(PATH, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(args.device)
    return model
