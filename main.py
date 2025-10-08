from train import run_training
from utils import load_data, load_problem
import argparse
import os
import torch
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_idx", help="config index", type=int, default=0)
    parser.add_argument("--dataset", help="data1001", default="data1001")
    parser.add_argument("--job", type=str, default="training")
    parser.add_argument("--continue_training", type=bool, default=False)

    # save related parameters
    parser.add_argument("--resultSaveFreq", type=int, default=1000)
    parser.add_argument("--resultPrintFreq", type=int, default=2)
    parser.add_argument("--float64", type=bool, default=False)

    return parser.parse_args()


def complete_args(cfg_file, init_args):
    init_args_dict = vars(init_args)
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    args_dict = {**init_args_dict, **cfg}
    args = argparse.Namespace(**args_dict)

    # Override
    if args.epochs < args.resultSaveFreq:
        args.resultSaveFreq = args.epochs
    args.model_id = f"{args.dataset}_cfg{args.cfg_idx}"
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Assert
    return args


def main(args):
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./data/results_summary'):
        os.makedirs('./data/results_summary')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./logs/run_training'):
        os.makedirs('./logs/run_training')

    if args.float64:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    if args.job == 'training':
        data = load_data(args)
        run_training(args, data)
    else:
        raise ValueError('Invalid job type')


if __name__ == '__main__':
    init_args = add_arguments()
    cgf_file = f"./cfg/{init_args.dataset}_{init_args.cfg_idx}"
    args = complete_args(cgf_file, init_args)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    main(args)
