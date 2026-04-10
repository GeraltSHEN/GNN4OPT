from .raw import GNNPolicy
from .lp import LPGNN


def get_model(args):
    if args.model == "raw":
        return GNNPolicy(
            emb_size=args.emb_size,
            cons_nfeats=cons_nfeats,
            var_nfeats=var_nfeats,
            n_layers=args.n_layers,
            gnn_backbone=args.gnn_backbone)
    elif args.model == "lp":
        return LPGNN(
            emb_size=args.emb_size,
            cons_nfeats=cons_nfeats,
            var_nfeats=var_nfeats,
            n_layers=args.n_layers,
            gnn_backbone=args.gnn_backbone)
    else:
        raise ValueError(f'{args.model} does not exist')
