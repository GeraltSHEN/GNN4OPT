import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml


BASE_DEFAULTS = {
    "max_samples_per_split": 128,
    "config_root": "./disjunctive_dual/cfg",
    "resume_model_dir": "",
    "model_suffix": "",
    "model_dir": "./disjunctive_dual/models",
    "log_dir": "./disjunctive_dual/logs",
    # model options
    "model": "raw",  # raw; holo; gumbel
    "hidden_channels": 64,
    "num_layers": 2,
    "n_breakings": 8,
    "num_heads": 1,
    "isab_num_inds": 50,
    "sym_break_layers": 2,
    "mp_layers": 2,
    "breaking_selector_model_path": None,
    # gumbel options
    "num_subgraphs": 2,
    "num_marked": 1,
    "gumbel_tau": 1.0,
    "gumbel_hard": True,
    "gumbel_use_noise": True,
    "gumbel_detach_marking": False,
    "gumbel_sym_break_layers": 2,
    "edge_nfeats": 1,
    "binarize_edge_features": False,
    "file_pattern": "sample_*.pkl",
    # training options
    "seed": 42,
    "epochs": 2,
    "lr": 1e-4,
    "batch_size": 8,
    "weight_decay": 5e-4,
    "loss_option": "ranking",
    "resume": False,
    "eval_every": 10000,
    "save_every": 10000,
    "print_every": 10000,
}

DATASET_DEFAULTS = {
    "set_cover": {
        "dataset_path": "legacy_code_generator/data/samples/setcover/500r_1000c_0.05d",
    }
}


def get_default_args(dataset: str):
    if dataset not in DATASET_DEFAULTS:
        raise NotImplementedError(f"Unsupported dataset '{dataset}'.")

    defaults = {**BASE_DEFAULTS, **DATASET_DEFAULTS[dataset]}
    return defaults


def write_config(defaults, config_root: Path, dataset: str, cfg_idx: int) -> Path:
    config_root.mkdir(parents=True, exist_ok=True)
    defaults["config_root"] = str(config_root)
    cfg_path = config_root / f"{dataset}_{cfg_idx}"
    with open(cfg_path, "w") as yaml_file:
        yaml.safe_dump(defaults, yaml_file, default_flow_style=False, sort_keys=False)
    print(f"Default Configuration file saved to {cfg_path}")
    return cfg_path


def parse_args(argv=None):
    """Parse CLI arguments for generating default configuration files."""
    parser = argparse.ArgumentParser(description="Generate default configuration files.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_DEFAULTS.keys()),
        help="Dataset name for which to generate defaults.",
    )
    parser.add_argument(
        "--cfg_idx",
        type=int,
        default=0,
        help="Configuration index appended to the generated file name.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    defaults = get_default_args(
        args.dataset,
    )
    write_config(defaults, Path(defaults['config_root']), args.dataset, args.cfg_idx)


if __name__ == "__main__":
    main(sys.argv[1:])
