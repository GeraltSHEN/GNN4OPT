import argparse
import sys

import yaml


def get_default_args(dataset):
    defaults = {}
    # logging related parameters
    defaults["config_root"] = "./cfg"
    defaults["resume_model_dir"] = ""
    defaults["model_suffix"] = ""
    defaults["model_dir"] = "./models"
    defaults["log_dir"] = "./logs"
    # dataset related parameters
    # layers related parameters
    defaults["model"] = "holo" # gcn; holo
    hidden_channels = 64
    defaults["hidden_channels"] = hidden_channels
    defaults["num_layers"] = 2
    defaults["dropout"] = 0.0
    defaults["n_breakings"] = 8  # for holo
    """This corresponds to the f_t in the paper"""
    defaults["symmetry_breaking_model"] = "power_method"  # power_method; gnn
    defaults["power"] = 2
    # training related parameters
    defaults["seed"] = 42
    defaults["epochs"] = 2
    defaults["lr"] = 1e-3
    defaults["batch_size"] = 32
    defaults["weight_decay"] = 5e-4
    defaults["out_dim"] = hidden_channels
    defaults["loss_option"] = "classification"
    defaults["resume"] = False

    if dataset == "set_cover":
        # logging related parameters
        defaults["eval_every"] = 1000
        defaults["save_every"] = 10000
        defaults["print_every"] = 1000
        # dataset related parameters
        defaults["dataset_path"] = "legacy_code_generator/data/samples/setcover/500r_1000c_0.05d"
        defaults["r"] = 1
        # layers related parameters
        # training related parameters
        defaults["sample_negatives"] = False
        pass
    else:
        raise NotImplementedError

    with open(f"./cfg/{dataset}_0", "w") as yaml_file:
        yaml.dump(defaults, yaml_file, default_flow_style=False)

    print(f"Default Configuration file saved to ./cfg/{dataset}_0")


def parse_args(argv=None):
    """Parse CLI arguments for generating default configuration files."""
    parser = argparse.ArgumentParser(description="Generate default configuration files.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["set_cover"],
        help="Dataset name for which to generate defaults."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    get_default_args(args.dataset)


if __name__ == "__main__":
    main(sys.argv[1:])
