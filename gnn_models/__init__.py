from .raw import GNNPolicy
from .lp import LPGNN


def get_model(args):
    if args.type == "raw":
        return GNNPolicy(
            emb_size=args.emb_size,
            cons_nfeats=2, # 2, 3
            var_nfeats=10, # 10, 12, 14
            n_layers=args.n_layers,
            gnn_backbone=args.gnn_backbone)
    elif args.type == "lp":
        return LPGNN(
            emb_size=args.emb_size,
            cons_nfeats=2, # 2, 3
            var_nfeats=12, # 12, 14, 16
            n_layers=args.n_layers,
            gnn_backbone=args.gnn_backbone)
    else:
        raise ValueError(f'{args.model} does not exist')
