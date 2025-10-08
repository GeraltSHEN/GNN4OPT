import torch

from models import (
    Classifier,
    Holo,
    GCNDataEncoder,
    PowerMethod,
    ProductTupleEncoder,
    SymmetryBreakingGNN,
)

def load_data(args):
    train_files = list(pathlib.Path(f'data/samples/{problem_folder}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'data/samples/{problem_folder}/valid').glob('sample_*.pkl'))



def load_model(args):
    data_encoder = GCNDataEncoder(
            in_channels=args.num_features,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    if args.model == "gcn":
        tuple_encoder = ProductTupleEncoder()
        in_dim = args.hidden_channels
    elif args.model == "holo":
        # TODO: why +1?
        in_dim = args.hidden_channels + 1
        if args.symmetry_breaking_model == "power_method":
            symmetry_breaking_model = PowerMethod(args.power, in_dim)
        elif args.symmetry_breaking_model == "gnn":
            symmetry_breaking_model = SymmetryBreakingGNN(in_dim, in_dim)
        else:
            raise ValueError(
                f"Unkown symmetry breaking model {args.symmetry_breaking_model}"
            )
        tuple_encoder = Holo(
            n_breakings=args.n_breakings,
            symmetry_breaking_model=symmetry_breaking_model,
        )
    else:
        raise NotImplementedError()
    out_dim = args.out_dim
    model = Classifier(
        data_encoder,
        tuple_encoder,
        in_dim=in_dim,
        out_dim=out_dim,
        linear_classifier=args.linear_classifier,
        train_head_only=args.pretrained_path is not None,
    )

    print(f"Model Architecture: \n{model}")
    model = model.to(args.device)
    return model


class CooSampler:
    def __init__(self, coos, values, batch_size: int, shuffle: bool = False):
        assert coos.size(1) == values.size(0)
        self.coos = coos
        self.values = values
        self.shuffle = shuffle
        n_groups, rem = divmod(self.values.size(0), batch_size)

        self.batch_sizes = [batch_size] * n_groups
        if rem > 0:
            self.batch_sizes.append(rem)

    def __len__(self):
        return len(self.batch_sizes)

    def __iter__(self):
        size = self.values.size(0)
        perm_coos = self.coos
        perm_values = self.values
        if self.shuffle:
            perm = torch.randperm(size)
            perm_coos = self.coos[:, perm]
            perm_values = self.values[perm]

        return iter(
            [
                (coos, values)
                for (coos, values) in zip(
                    torch.split(perm_coos, self.batch_sizes, dim=1),
                    torch.split(perm_values, self.batch_sizes, dim=0),
                )
            ]
        )


def get_optimizer(args, model):
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # ... Handle pretrained model optimizer state dict if needed
    return optimizer