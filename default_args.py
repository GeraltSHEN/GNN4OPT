import argparse
import sys
from pathlib import Path

import yaml
from distutils.util import strtobool

NONE_ALIASES = {"none", "null", "nil"}


BASE_DEFAULTS = {
    "max_samples_per_split": 128,
    "config_root": "./cfg",
    "resume_model_dir": "",
    "model_suffix": "",
    "model_dir": "./models",
    "log_dir": "./logs",
    # model options
    "model": "raw",  # raw; holo
    "hidden_channels": 64,
    "num_layers": 2,
    "n_breakings": 8,
    "num_heads": 1,
    "isab_num_inds": 50,
    "sym_break_layers": 2,
    "mp_layers": 2,
    "use_set_transformer": True,
    "breaking_selector_model_path": None,
    "edge_nfeats": 1,
    "use_default_features": False,
    "remove_bad_candidates": False,
    "file_pattern": "sample_*.pkl",
    # training options
    "seed": 42,
    "epochs": 1,
    "lr": 1e-4,
    "batch_size": 8,
    "weight_decay": 5e-4,
    "loss_option": "LambdaNDCGLoss1",
    "tier1_ub": 0.0,
    "relevance_type": "linear",
    "resume": False,
    "eval_every": 100000,
    "save_every": 100000,
    "print_every": 100000,
}

DATASET_DEFAULTS = {
    "set_cover": {
        "dataset_path": "legacy_code_generator/data/samples/setcover/500r_1000c_0.05d",
    },
    "cauctions": {
        "dataset_path": "legacy_code_generator/data/samples/cauctions/100_500",
    },
    "facilities": {
        "dataset_path": "legacy_code_generator/data/samples/facilities/100_100_5",
    },
    "indset": {
        "dataset_path": "legacy_code_generator/data/samples/indset/500_4",
    }
}


def get_default_args(dataset: str):
    if dataset not in DATASET_DEFAULTS:
        raise NotImplementedError(f"Unsupported dataset '{dataset}'.")

    defaults = {**BASE_DEFAULTS, **DATASET_DEFAULTS[dataset]}
    return defaults


def is_none_token(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in NONE_ALIASES)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    try:
        return bool(strtobool(str(value)))
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Expected a boolean value for '{value}'.") from exc


def register_override_arguments(parser: argparse.ArgumentParser, defaults: dict):
    """Dynamically register CLI flags for each configurable default key."""

    def none_or(parser_fn):
        def wrapper(raw):
            if is_none_token(raw):
                return None
            return parser_fn(raw)

        return wrapper

    for key, default_value in defaults.items():
        # Skip required positional arguments that are handled separately
        if key in {"dataset", "cfg_idx"}:
            continue

        arg_kwargs = {
            "default": argparse.SUPPRESS,
            "help": f"Override default for '{key}' (default: {default_value}).",
        }

        if isinstance(default_value, bool):
            arg_kwargs["type"] = none_or(parse_bool)
        elif isinstance(default_value, int):
            arg_kwargs["type"] = none_or(int)
        elif isinstance(default_value, float):
            arg_kwargs["type"] = none_or(float)
        else:
            arg_kwargs["type"] = none_or(str)

        parser.add_argument(f"--{key}", **arg_kwargs)


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
    # Allow overriding any default values via CLI
    all_defaults = {**BASE_DEFAULTS}
    for dataset_defaults in DATASET_DEFAULTS.values():
        all_defaults.update(dataset_defaults)
    register_override_arguments(parser, all_defaults)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    defaults = get_default_args(
        args.dataset,
    )
    # Apply any CLI overrides (only for flags that were explicitly provided)
    for key in defaults.keys():
        if hasattr(args, key):
            defaults[key] = getattr(args, key)

    write_config(defaults, Path(defaults["config_root"]), args.dataset, args.cfg_idx)


if __name__ == "__main__":
    main(sys.argv[1:])
