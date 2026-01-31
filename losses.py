import torch as _torch


def batch_pairs(x: _torch.Tensor) -> _torch.Tensor:
    """Returns a pair matrix (copied from pytorchltr.utils.tensor_operations)."""
    if x.dim() == 2:
        x = x.reshape((x.shape[0], x.shape[1], 1))

    x_ij = _torch.repeat_interleave(x, x.shape[1], dim=2)
    x_ji = _torch.repeat_interleave(x.permute(0, 2, 1), x.shape[1], dim=1)

    return _torch.stack([x_ij, x_ji], dim=3)


def mask_padded_values(xs: _torch.FloatTensor, n: _torch.LongTensor,
                       mask_value: float = -float('inf'),
                       mutate: bool = False):
    """Turns padded values into given mask value."""
    mask = _torch.repeat_interleave(
        _torch.arange(xs.shape[1], device=xs.device).reshape((1, xs.shape[1])),
        xs.shape[0], dim=0)
    n_mask = _torch.repeat_interleave(
        n.reshape((n.shape[0], 1)), xs.shape[1], dim=1)
    if not mutate:
        xs = xs.clone()
    xs[mask >= n_mask] = mask_value
    return xs


def tiebreak_argsort(
        x: _torch.FloatTensor,
        descending: bool = True,
        generator: _torch.Generator = None) -> _torch.LongTensor:
    """Computes a per-row argsort of matrix x with random tiebreaks."""
    rng_kwargs = {"generator": generator} if generator is not None else {}
    p = _torch.randperm(x.shape[1], device=x.device, **rng_kwargs)
    return p[_torch.argsort(x[:, p], descending=descending)]


def rank_by_score(
        scores: _torch.FloatTensor,
        n: _torch.LongTensor,
        generator: _torch.Generator = None) -> _torch.LongTensor:
    """Sorts scores in decreasing order with padded docs placed last."""
    if scores.dim() == 3:
        scores = scores.reshape((scores.shape[0], scores.shape[1]))
    return tiebreak_argsort(mask_padded_values(scores, n), generator=generator)


_batch_pairs = batch_pairs
_rank_by_score = rank_by_score


class TopTierAverageSoftmaxLoss(_torch.nn.Module):
    # DEPRECATED
    r"""Averaged multi-positive softmax loss on the top tier.

    Implements the loss
        L = log( sum_j exp(s_j) ) - log( (1 / |T_1|) * sum_{i in T_1} exp(s_i) )
    where T_1 contains the documents in the highest relevance tier.

    Shape:
        - input scores: :math:`(N, list\_size)` or `(N, list_size, 1)`
        - input relevance: :math:`(N, list\_size)` or `(N, list_size, 1)`
        - input n: :math:`(N)`
        - output: :math:`(N)`
    """
    def forward(self, scores: _torch.FloatTensor,
                relevance: _torch.FloatTensor,
                n: _torch.LongTensor) -> _torch.FloatTensor:
        if relevance.ndimension() == 3:
            relevance = relevance.reshape(
                (relevance.shape[0], relevance.shape[1]))
        if scores.ndimension() == 3:
            scores = scores.reshape((scores.shape[0], scores.shape[1]))

        _, list_size = scores.shape
        device = scores.device
        valid_mask = (_torch.arange(list_size, device=device)
                      .unsqueeze(0) < n.unsqueeze(1))

        positive_mask = (relevance > 0) & valid_mask
        positive_counts = positive_mask.sum(dim=1)

        masked_scores = scores.masked_fill(~valid_mask, -float("inf"))
        log_denom = _torch.logsumexp(masked_scores, dim=1)

        positive_scores = scores.masked_fill(~positive_mask, -float("inf"))
        log_positive_sum = _torch.logsumexp(positive_scores, dim=1)
        safe_counts = positive_counts.clamp(min=1).float()
        log_positive_mean = log_positive_sum - _torch.log(safe_counts)

        loss = log_denom - log_positive_mean
        loss = _torch.where(positive_counts > 0, loss,
                            _torch.zeros_like(loss))
        return loss


class NormalizedPairwiseLogisticLoss(_torch.nn.Module):
    # DEPRECATED
    r"""Normalized pairwise logistic loss.

    Implements the loss
        L = (1 / |R|) * sum_{i in R} log(1 + sum_{j not in R} exp(-sigma (s_i - s_j))).
    Here R is the set of indices with the highest relevance label (assuming two
    distinct relevance levels) for a given query, and only comparisons between R
    and its complement are considered.

    Shape:
        - input scores: :math:`(N, list_size)` or `(N, list_size, 1)`
        - input relevance: :math:`(N, list_size)` or `(N, list_size, 1)`
        - input n: :math:`(N)`
        - output: :math:`(N)`
    """
    def __init__(self, sigma: float = 1.0, epsilon: float = 0.0):
        """
        Args:
            sigma: Steepness of the logistic curve.
        """
        super().__init__()
        self.sigma = sigma
        self.epsilon = epsilon

    def forward(self, scores: _torch.FloatTensor,
                relevance: _torch.FloatTensor,
                n: _torch.LongTensor) -> _torch.FloatTensor:
        # Reshape to match the layout used by PairwiseLogisticLoss.
        if relevance.ndimension() == 2:
            relevance = relevance.reshape(
                (relevance.shape[0], relevance.shape[1], 1))
        if scores.ndimension() == 2:
            scores = scores.reshape((scores.shape[0], scores.shape[1], 1))

        score_pairs = batch_pairs(scores)
        # Same core difference computation as PairwiseLogisticLoss.
        score_pair_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]

        batch_size, list_size = scores.shape[:2]
        device = scores.device

        relevance_flat = relevance.reshape((batch_size, list_size))
        valid_mask = (_torch.arange(list_size, device=device)
                      .unsqueeze(0) < n.unsqueeze(1))
        relevance_flat = relevance_flat.float()
        masked_relevance = relevance_flat.masked_fill(
            ~valid_mask, -float("inf"))
        max_relevance = masked_relevance.max(dim=1, keepdim=True).values
        # Two-tier assumption: top tier is exactly the max relevance.
        top_mask = valid_mask & (relevance_flat == max_relevance)
        non_top_mask = valid_mask & ~top_mask

        # Difference from PairwiseLogisticLoss: only compare i in R to j not in R.
        pair_mask = (top_mask[:, :, None] & non_top_mask[:, None, :]).float()

        # Difference from PairwiseLogisticLoss: aggregate exp terms per i, then log2.
        exp_terms = _torch.exp(-self.sigma * score_pair_diffs) * pair_mask
        summed_exp = exp_terms.sum(dim=2)
        log_terms = _torch.log2(1.0 + summed_exp)

        # Difference from PairwiseLogisticLoss: normalize by |R| and ignore padding.
        top_counts = top_mask.sum(dim=1).clamp(min=1)
        loss = (log_terms * top_mask.float()).sum(dim=1) / top_counts

        return loss


class LambdaLoss(_torch.nn.Module):
    """LambdaLoss."""
    def __init__(self, sigma: float = 1.0):
        """
        Args:
            sigma: Steepness of the logistic curve.
        """
        super().__init__()
        self.sigma = sigma

    def _loss_per_doc_pair(self, score_pairs: _torch.FloatTensor,
                           rel_pairs: _torch.LongTensor,
                           n: _torch.LongTensor) -> _torch.FloatTensor:
        """Computes a loss on given score pairs and relevance pairs.

        Args:
            score_pairs: A tensor of shape (batch_size, list_size,
                list_size, 2), where each entry (:, i, j, :) indicates a pair
                of scores for doc i and j.
            rel_pairs: A tensor of shape (batch_size, list_size, list_size, 2),
                where each entry (:, i, j, :) indicates the relevance
                for doc i and j.
            n: A batch of per-query number of documents (for padding purposes).

        Returns:
            A tensor of shape (batch_size, list_size, list_size) with a loss
            per document pair.
        """
        raise NotImplementedError

    def _loss_reduction(self,
                        loss_pairs: _torch.FloatTensor) -> _torch.FloatTensor:
        """Reduces the paired loss to a per sample loss.

        Args:
            loss_pairs: A tensor of shape (batch_size, list_size, list_size)
                where each entry indicates the loss of doc pair i and j.

        Returns:
            A tensor of shape (batch_size) indicating the loss per training
            sample.
        """
        return loss_pairs.view(loss_pairs.shape[0], -1).sum(1)

    def forward(self, scores: _torch.FloatTensor, relevance: _torch.LongTensor,
                n: _torch.LongTensor) -> _torch.FloatTensor:
        """Computes the loss for given batch of samples.

        Args:
            scores: A batch of per-query-document scores.
            relevance: A batch of per-query-document relevance labels.
            n: A batch of per-query number of documents (for padding purposes).
        """
        # Reshape relevance if necessary.
        if relevance.ndimension() == 2:
            relevance = relevance.reshape(
                (relevance.shape[0], relevance.shape[1], 1))
        if scores.ndimension() == 2:
            scores = scores.reshape((scores.shape[0], scores.shape[1], 1))

        # Compute ranking and sort scores and relevance
        ranking = _rank_by_score(scores, n)
        ranking = ranking.view((ranking.shape[0], ranking.shape[1], 1))
        scores = _torch.gather(scores, 1, ranking)
        relevance = _torch.gather(relevance, 1, ranking)

        # Compute pairwise differences for scores and relevances.
        score_pairs = _batch_pairs(scores)
        rel_pairs = _batch_pairs(relevance)

        # Compute loss per doc pair.
        loss_pairs = self._loss_per_doc_pair(score_pairs, rel_pairs, n)

        # Mask out padded documents per query in the batch
        n_grid = n[:, None, None].repeat(1, score_pairs.shape[1],
                                         score_pairs.shape[2])
        arange = _torch.arange(score_pairs.shape[1],
                               device=score_pairs.device)
        range_grid = _torch.max(*_torch.meshgrid([arange, arange]))
        range_grid = range_grid[None, :, :].repeat(n.shape[0], 1, 1)
        loss_pairs[n_grid <= range_grid] = 0.0

        # Reduce final list loss from per doc pair loss to a per query loss.
        loss = self._loss_reduction(loss_pairs)

        # Return loss
        return loss


class TierNormalizedLambdaARP2(LambdaLoss):
    # DEPRECATED
    r"""
    Tier-normalized ARP Loss 2.

    .. math::
        l(\mathbf{s}, \mathbf{y})
        = \sum_{a < b} \frac{1}{|T_a||T_b|}\sum_{i \in T_a}\sum_{j \in T_b}
        |y_i - y_j| \log_2(1 + e^{-\sigma(s_i - s_j)})
    """
    def _loss_per_doc_pair(self, score_pairs, rel_pairs, n):
        score_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_diffs = rel_pairs[:, :, :, 0] - rel_pairs[:, :, :, 1]
        loss = _torch.log2(1.0 + _torch.exp(-self.sigma * score_diffs))
        loss[rel_diffs <= 0] = 0.0

        # CHANGED: compute per-tier document counts for normalization.
        rel_per_doc = rel_pairs[:, :, 0, 0]
        batch_size, list_size = rel_per_doc.shape
        device = score_pairs.device
        valid_mask = (_torch.arange(list_size, device=device)
                      .unsqueeze(0) < n.unsqueeze(1))
        valid_labels = rel_per_doc[valid_mask]
        num_labels = int(valid_labels.max().item()) + 1 if valid_labels.numel() > 0 else 1
        label_counts = _torch.zeros((batch_size, num_labels),
                                    device=device, dtype=_torch.float)
        if valid_labels.numel() > 0:
            batch_indices = _torch.arange(batch_size, device=device).unsqueeze(1)
            batch_indices = batch_indices.expand(-1, list_size)[valid_mask]
            label_counts.index_put_((batch_indices, valid_labels),
                                    _torch.ones_like(valid_labels, dtype=_torch.float),
                                    accumulate=True)
        label_counts = label_counts.clamp(min=1.0)

        # CHANGED: normalize each pairwise term by |T_a||T_b|.
        rel_i = rel_pairs[:, :, :, 0].clamp(max=num_labels - 1)
        rel_j = rel_pairs[:, :, :, 1].clamp(max=num_labels - 1)
        label_counts_expanded = label_counts[:, None, None, :].expand(batch_size, list_size, list_size, num_labels)
        count_i = label_counts_expanded.gather(3, rel_i.unsqueeze(-1)).squeeze(-1)
        count_j = label_counts_expanded.gather(3, rel_j.unsqueeze(-1)).squeeze(-1)
        tier_weight = 1.0 / (count_i * count_j)

        return rel_diffs * loss * tier_weight


class LiPO(LambdaLoss):
    r"""LiPO loss."""

    def _loss_per_doc_pair(self, score_pairs, rel_pairs, n):
        score_diffs = score_pairs[:, :, :, 0] - score_pairs[:, :, :, 1]
        rel_i = rel_pairs[:, :, :, 0].float()
        rel_j = rel_pairs[:, :, :, 1].float()

        # CHANGED: LiPO weighting with gain differences and inverse discount gaps.
        gain_i = _torch.pow(2.0, rel_i) - 1.0
        gain_j = _torch.pow(2.0, rel_j) - 1.0

        batch_size, list_size, _ = rel_i.shape
        device = score_pairs.device
        positions = _torch.arange(list_size, device=device,
                                  dtype=_torch.float) + 1.0
        inv_discount = 1.0 / _torch.log2(1.0 + positions)
        inv_d_i = inv_discount.view(1, list_size, 1).expand(batch_size,
                                                            list_size,
                                                            list_size)
        inv_d_j = inv_discount.view(1, 1, list_size).expand(batch_size,
                                                            list_size,
                                                            list_size)

        weight = _torch.abs(gain_i - gain_j) * _torch.abs(inv_d_i - inv_d_j)

        loss = _torch.log2(1.0 + _torch.exp(-self.sigma * score_diffs))
        loss = loss * (rel_i > rel_j).float()

        return loss * weight

class TierAwarePairwiseLogisticLoss(_torch.nn.Module):
    r"""Tier-aware pairwise logistic loss.

    For a query with valid documents i, j and scores s_i, s_j, define tiers:
    tier1: y_i > 0, tier2: y_i <= 0. Let n1, n2 be tier counts and
    n_i = n1 if y_i > 0 else n2. The pairwise weight is
    alpha_ij = c_ij / (n_i * n_j), where c_11 = c_12 = c_21 = 0.3 and
    c_22 = 0.1. 
    The loss is:

        L = sum_{i != j} [
              2 * alpha_ij * log2(1 + exp(-sigma * (s_i - s_j))) * I[y_i > y_j]
            + 0.5 * alpha_ij * (log2(1 + exp(-sigma * (s_i - s_j)))
                               +log2(1 + exp( sigma * (s_i - s_j))))
              * I[y_i = y_j]
            ]
    """
    def __init__(self,
                 sigma: float = 1.0,
                 c_11: float = 0.3,
                 c_12: float = 0.3,
                 c_21: float = 0.3,
                 c_22: float = 0.1):
        super().__init__()
        self.sigma = sigma
        self.c_11 = c_11
        self.c_12 = c_12
        self.c_21 = c_21
        self.c_22 = c_22

    def forward(self, scores: _torch.FloatTensor,
                relevance: _torch.FloatTensor,
                n: _torch.LongTensor) -> _torch.FloatTensor:
        if relevance.ndimension() == 3:
            relevance = relevance.reshape(
                (relevance.shape[0], relevance.shape[1]))
        if scores.ndimension() == 3:
            scores = scores.reshape((scores.shape[0], scores.shape[1]))

        batch_size, list_size = scores.shape
        device = scores.device

        valid_mask = (_torch.arange(list_size, device=device)
                      .unsqueeze(0) < n.unsqueeze(1))
        y = relevance.float()

        tier1_mask = (y > 0) & valid_mask
        tier2_mask = (~tier1_mask) & valid_mask

        tier1_counts = tier1_mask.sum(dim=1).clamp(min=1).float()
        tier2_counts = tier2_mask.sum(dim=1).clamp(min=1).float()

        count_i = _torch.where(tier1_mask, tier1_counts.unsqueeze(1),
                               tier2_counts.unsqueeze(1))
        count_i = count_i * valid_mask.float()
        count_j = count_i

        denom_ij = count_i[:, :, None] * count_j[:, None, :]
        denom_ij = _torch.where(denom_ij > 0.0, denom_ij,
                                _torch.ones_like(denom_ij))

        tier1_i = tier1_mask[:, :, None].float()
        tier1_j = tier1_mask[:, None, :].float()
        tier2_i = tier2_mask[:, :, None].float()
        tier2_j = tier2_mask[:, None, :].float()

        numer_ij = (self.c_11 * tier1_i * tier1_j
                    + self.c_12 * tier1_i * tier2_j
                    + self.c_21 * tier2_i * tier1_j
                    + self.c_22 * tier2_i * tier2_j)

        alpha_ij = numer_ij / denom_ij

        score_diffs = scores[:, :, None] - scores[:, None, :]
        loss_forward = _torch.log2(1.0 + _torch.exp(-self.sigma * score_diffs))
        loss_backward = _torch.log2(1.0 + _torch.exp(self.sigma * score_diffs))

        if list_size > 1:
            diag_mask = ~_torch.eye(list_size, dtype=_torch.bool, device=device)
            pair_mask = (valid_mask[:, :, None] & valid_mask[:, None, :]
                         & diag_mask.unsqueeze(0))
        else:
            pair_mask = valid_mask[:, :, None] & valid_mask[:, None, :]

        y_i = y[:, :, None]
        y_j = y[:, None, :]
        greater_mask = (y_i > y_j) & pair_mask
        equal_mask = (y_i == y_j) & pair_mask

        loss = (2.0 * alpha_ij * loss_forward * greater_mask.float()
                + 0.5 * alpha_ij * (loss_forward + loss_backward)
                * equal_mask.float())

        return loss.sum(dim=(1, 2))


class NCE(_torch.nn.Module):
    r"""Implementation of ListNet ranking loss (or from "Noise Contrastive Alignment of Language Models with Explicit Rewards")
    L = - sum_{i \in Candidate Set} exp(sigma y_i) / (sum_j exp(sigma y_j)) 
                log(exp(sigma s_i) / (sum_j exp(sigma s_j)))
    """
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, scores: _torch.FloatTensor, relevance: _torch.FloatTensor, n: _torch.LongTensor) -> _torch.FloatTensor:
        if relevance.ndimension() == 3:
            relevance = relevance.reshape(
                (relevance.shape[0], relevance.shape[1]))
        if scores.ndimension() == 3:
            scores = scores.reshape((scores.shape[0], scores.shape[1]))

        _, list_size = scores.shape
        device = scores.device
        valid_mask = (_torch.arange(list_size, device=device)
                      .unsqueeze(0) < n.unsqueeze(1))

        scaled_scores = scores * self.sigma
        scaled_relevance = relevance * self.sigma

        masked_scores = scaled_scores.masked_fill(~valid_mask, -float("inf"))
        masked_relevance = scaled_relevance.masked_fill(~valid_mask,
                                                        -float("inf"))

        log_probs = _torch.log_softmax(masked_scores, dim=1)
        log_probs = log_probs.masked_fill(~valid_mask, 0.0)
        target_probs = _torch.softmax(masked_relevance, dim=1)
        target_probs = target_probs.masked_fill(~valid_mask, 0.0)

        loss = -(target_probs * log_probs).sum(dim=1)
        return loss
