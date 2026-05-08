import os
import argparse
import multiprocessing as mp
import pickle
import glob
import numpy as np
import shutil
import gzip

import pyscipopt as scip
import utilities


DUAL_OPTION_DEFAULT = 1
LEGACY_TARGET_KEY = "top8_regression_targets"
OPTION1_TOPK_TARGET_KEY = "top8_regression_targets_option1"
OPTION1_ALL_TARGET_KEY = "all_regression_targets_option1"


def select_topk_local_positions(candidate_scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0 or int(k) <= 0:
        return np.empty((0,), dtype=np.int64)
    k_eff = int(min(max(int(k), 0), scores.size))
    top_positions = np.argsort(-scores)[:k_eff]
    return top_positions.astype(np.int64, copy=False)


def build_option1_targets(
    action_set,
    scores,
    canddowns,
    candups,
    cutoffbound: float,
    parent_obj: float,
    top_k: int,
):
    if canddowns is None or candups is None:
        return None, None

    action_set = np.asarray(action_set, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    canddowns = np.asarray(canddowns, dtype=np.float32).reshape(-1)
    candups = np.asarray(candups, dtype=np.float32).reshape(-1)
    if not (action_set.size == scores.size == canddowns.size == candups.size):
        raise ValueError(
            "Mismatched candidate payload sizes: "
            f"action_set={action_set.size}, scores={scores.size}, "
            f"canddowns={canddowns.size}, candups={candups.size}"
        )

    all_positions = np.arange(action_set.size, dtype=np.int64)
    all_obj = np.stack((canddowns, candups), axis=1).astype(np.float32, copy=False)

    effective_top_k = action_set.size if int(top_k) <= 0 else int(top_k)
    top_positions = select_topk_local_positions(scores, effective_top_k)
    top_candidates = action_set[top_positions]
    top_scores = scores[top_positions]
    top_obj = all_obj[top_positions]

    option_one_top = {
        "candidate_positions": top_positions.copy(),
        "candidate_indices": top_candidates.copy(),
        "scores": top_scores.copy(),
        "sample_scores": top_scores.copy(),
        "obj": top_obj.copy(),
        "parent_obj": np.float32(parent_obj),
        "cutoffbound": np.float32(cutoffbound),
        "sample_cutoffbound": np.float32(cutoffbound),
        "dual_option": int(DUAL_OPTION_DEFAULT),
    }
    option_one_all = {
        "candidate_positions": all_positions.copy(),
        "candidate_indices": action_set.copy(),
        "scores": scores.copy(),
        "sample_scores": scores.copy(),
        "obj": all_obj.copy(),
        "parent_obj": np.float32(parent_obj),
        "cutoffbound": np.float32(cutoffbound),
        "sample_cutoffbound": np.float32(cutoffbound),
        "dual_option": int(DUAL_OPTION_DEFAULT),
    }
    return option_one_top, option_one_all


class SamplingAgent(scip.Branchrule):

    def __init__(self, episode, instance, seed, out_queue, exploration_policy, query_expert_prob, dual_top_k, out_dir, follow_expert=True):
        self.episode = episode
        self.instance = instance
        self.seed = seed
        self.out_queue = out_queue
        self.exploration_policy = exploration_policy
        self.query_expert_prob = query_expert_prob
        self.out_dir = out_dir
        self.dual_top_k = int(max(0, dual_top_k))
        self.follow_expert = follow_expert

        self.rng = np.random.RandomState(seed)
        self.new_node = True
        self.sample_counter = 0
        self.warned_missing_cand_obj = False

    def branchinit(self):
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):

        if self.model.getNNodes() == 1:
            # initialize root buffer for Khalil features extraction
            utilities.extract_khalil_variable_features(self.model, [], self.khalil_root_buffer)

        # once in a while, also run the expert policy and record the (state, action) pair
        query_expert = self.rng.rand() < self.query_expert_prob
        if query_expert:
            cutoffbound = float(self.model.getCutoffbound())
            state = utilities.extract_state(self.model, cutoffbound=cutoffbound)
            cands, *_ = self.model.getPseudoBranchCands()
            state_khalil = utilities.extract_khalil_variable_features(self.model, cands, self.khalil_root_buffer)

            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, canddowns, candups, _npriocands, bestcand = self.model.getVanillafullstrongData()

            assert result == scip.SCIP_RESULT.DIDNOTRUN
            assert all([c1.getCol().getLPPos() == c2.getCol().getLPPos() for c1, c2 in zip(cands, cands_)])

            action_set = [c.getCol().getLPPos() for c in cands]
            expert_action = action_set[bestcand]

            data = [state, state_khalil, expert_action, action_set, scores, cutoffbound]
           
            # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
            if not any([s < 0 for s in scores]):
                sample_payload = {
                    'episode': self.episode,
                    'instance': self.instance,
                    'seed': self.seed,
                    'node_number': self.model.getCurrentNode().getNumber(),
                    'node_depth': self.model.getCurrentNode().getDepth(),
                    'data': data,
                }

                option_one_top, option_one_all = build_option1_targets(
                    action_set=action_set,
                    scores=scores,
                    canddowns=canddowns,
                    candups=candups,
                    cutoffbound=float(cutoffbound),
                    parent_obj=float(self.model.getLPObjVal()),
                    top_k=self.dual_top_k,
                )
                if option_one_top is not None:
                    sample_payload[OPTION1_TOPK_TARGET_KEY] = option_one_top
                    sample_payload[LEGACY_TARGET_KEY] = option_one_top
                    sample_payload[OPTION1_ALL_TARGET_KEY] = option_one_all
                elif not self.warned_missing_cand_obj:
                    print(
                        "[WARN] getVanillafullstrongData() did not return canddowns/candups; "
                        "option1 labels are not embedded in samples. "
                        "Use patched PySCIPOpt or run 03_generate_lp_labels.py."
                    )
                    self.warned_missing_cand_obj = True

                filename = f'{self.out_dir}/sample_{self.episode}_{self.sample_counter}.pkl'
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(sample_payload, f)

                self.out_queue.put({
                    'type': 'sample',
                    'episode': self.episode,
                    'instance': self.instance,
                    'seed': self.seed,
                    'node_number': self.model.getCurrentNode().getNumber(),
                    'node_depth': self.model.getCurrentNode().getDepth(),
                    'filename': filename,
                })

                self.sample_counter += 1

        # if exploration and expert policies are the same, prevent running it twice
        if not query_expert or (not self.follow_expert and self.exploration_policy != 'vanillafullstrong'):
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)

        # apply 'vanillafullstrong' branching decision if needed
        if query_expert and self.follow_expert or self.exploration_policy == 'vanillafullstrong':
            assert result == scip.SCIP_RESULT.DIDNOTRUN
            cands, _scores, _canddowns, _candups, _npriocands, bestcand = self.model.getVanillafullstrongData()
            self.model.branchVar(cands[bestcand])
            result = scip.SCIP_RESULT.BRANCHED

        return {"result": result}


def make_samples(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """

    while True:
        episode, instance, seed, exploration_policy, query_expert_prob, time_limit, dual_top_k, out_dir = in_queue.get()
        print(f'[w {os.getpid()}] episode {episode}, seed {seed}, processing instance \'{instance}\'...')

        m = scip.Model()
        m.setIntParam('display/verblevel', 0)
        m.readProblem(f'{instance}')
        utilities.init_scip_params(m, seed=seed)
        m.setIntParam('timing/clocktype', 2)
        m.setRealParam('limits/time', time_limit)

        branchrule = SamplingAgent(
            episode=episode,
            instance=instance,
            seed=seed,
            out_queue=out_queue,
            exploration_policy=exploration_policy,
            query_expert_prob=query_expert_prob,
            dual_top_k=dual_top_k,
            out_dir=out_dir)

        m.includeBranchrule(
            branchrule=branchrule,
            name="Sampling branching rule", desc="",
            priority=666666, maxdepth=-1, maxbounddist=1)

        m.setBoolParam('branching/vanillafullstrong/integralcands', True)
        m.setBoolParam('branching/vanillafullstrong/scoreall', True)
        m.setBoolParam('branching/vanillafullstrong/collectscores', True)
        m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
        m.setBoolParam('branching/vanillafullstrong/idempotent', True)

        out_queue.put({
            'type': 'start',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })

        m.optimize()
        m.freeProb()

        print(f"[w {os.getpid()}] episode {episode} done, {branchrule.sample_counter} samples")

        out_queue.put({
            'type': 'done',
            'episode': episode,
            'instance': instance,
            'seed': seed,
        })


def send_orders(orders_queue, instances, seed, exploration_policy, query_expert_prob, time_limit, dual_top_k, out_dir):
    """
    Continuously send sampling orders to workers (relies on limited
    queue capacity).

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Queue to which to send orders.
    instances : list
        Instance file names from which to sample episodes.
    seed : int
        Random seed for reproducibility.
    exploration_policy : str
        Branching strategy for exploration.
    query_expert_prob : float in [0, 1]
        Probability of running the expert strategy and collecting samples.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    out_dir: str
        Output directory in which to write samples.
    """
    rng = np.random.RandomState(seed)

    episode = 0
    while True:
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        orders_queue.put([episode, instance, seed, exploration_policy, query_expert_prob, time_limit, dual_top_k, out_dir])
        episode += 1


def collect_samples(instances, out_dir, rng, n_samples, n_jobs,
                    exploration_policy, query_expert_prob, time_limit, dual_top_k):
    """
    Runs branch-and-bound episodes on the given set of instances, and collects
    randomly (state, action) pairs from the 'vanilla-fullstrong' expert
    brancher.

    Parameters
    ----------
    instances : list
        Instance files from which to collect samples.
    out_dir : str
        Directory in which to write samples.
    rng : numpy.random.RandomState
        A random number generator for reproducibility.
    n_samples : int
        Number of samples to collect.
    n_jobs : int
        Number of jobs for parallel sampling.
    exploration_policy : str
        Exploration policy (branching rule) for sampling.
    query_expert_prob : float in [0, 1]
        Probability of using the expert policy and recording a (state, action)
        pair.
    time_limit : float in [0, 1e+20]
        Maximum running time for an episode, in seconds.
    """
    os.makedirs(out_dir, exist_ok=True)

    # start workers
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    tmp_samples_dir = f'{out_dir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # start dispatcher
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), exploration_policy, query_expert_prob, time_limit, dual_top_k, tmp_samples_dir),
            daemon=True)
    dispatcher.start()

    # record answers and write samples
    buffer = {}
    current_episode = 0
    i = 0
    in_buffer = 0
    while i < n_samples:
        sample = answers_queue.get()

        # add received sample to buffer
        if sample['type'] == 'start':
            buffer[sample['episode']] = []
        else:
            buffer[sample['episode']].append(sample)
            if sample['type'] == 'sample':
                in_buffer += 1

        # if any, write samples from current episode
        while current_episode in buffer and buffer[current_episode]:
            samples_to_write = buffer[current_episode]
            buffer[current_episode] = []

            for sample in samples_to_write:

                # if no more samples here, move to next episode
                if sample['type'] == 'done':
                    del buffer[current_episode]
                    current_episode += 1

                # else write sample
                else:
                    os.rename(sample['filename'], f'{out_dir}/sample_{i+1}.pkl')
                    in_buffer -= 1
                    i += 1
                    print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                    # early stop dispatcher (hard)
                    if in_buffer + i >= n_samples and dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")

                    # as soon as enough samples are collected, stop
                    if i == n_samples:
                        buffer = {}
                        break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='default',
        help='Dataset suffix. Special values: "default" -> base path, "debug" -> *_debug path + reduced sample counts.',
    )
    parser.add_argument(
        '--dual_top_k',
        help='Number of top-scored candidates saved under top8_regression_targets_option1 (<=0 means all).',
        type=int,
        default=8,
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-j', '--njobs',
        help='Number of parallel jobs.',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    mode = str(args.mode).strip()
    is_debug = mode == 'debug'
    if mode in {'', 'default'}:
        sample_problem_dir = args.problem
    elif is_debug:
        sample_problem_dir = f"{args.problem}_debug"
    else:
        sample_problem_dir = f"{args.problem}_{mode}"

    print(f"seed {args.seed}")

    train_size = 100000 # 100000
    valid_size = 20000 # 20000
    test_size = 20000 # 20000
    if is_debug:
        train_size //= 10
        valid_size //= 10
        test_size //= 10
    exploration_strategy = 'pscost'
    node_record_prob = 0.05
    time_limit = 3600

    if args.problem == 'setcover':
        instances_train = glob.glob('data/instances/setcover/train_500r_1000c_0.05d/*.lp')
        instances_valid = glob.glob('data/instances/setcover/valid_500r_1000c_0.05d/*.lp')
        instances_test = glob.glob('data/instances/setcover/test_500r_1000c_0.05d/*.lp')
        out_dir = f'data/samples/{sample_problem_dir}/500r_1000c_0.05d'

    elif args.problem == 'cauctions':
        instances_train = glob.glob('data/instances/cauctions/train_100_500/*.lp')
        instances_valid = glob.glob('data/instances/cauctions/valid_100_500/*.lp')
        instances_test = glob.glob('data/instances/cauctions/test_100_500/*.lp')
        out_dir = f'data/samples/{sample_problem_dir}/100_500'

    elif args.problem == 'indset':
        instances_train = glob.glob('data/instances/indset/train_500_4/*.lp')
        instances_valid = glob.glob('data/instances/indset/valid_500_4/*.lp')
        instances_test = glob.glob('data/instances/indset/test_500_4/*.lp')
        out_dir = f'data/samples/{sample_problem_dir}/500_4'

    elif args.problem == 'facilities':
        instances_train = glob.glob('data/instances/facilities/train_100_100_5/*.lp')
        instances_valid = glob.glob('data/instances/facilities/valid_100_100_5/*.lp')
        instances_test = glob.glob('data/instances/facilities/test_100_100_5/*.lp')
        out_dir = f'data/samples/{sample_problem_dir}/100_100_5'
        time_limit = 600

    else:
        raise NotImplementedError

    print(f"{len(instances_train)} train instances for {train_size} samples")
    print(f"{len(instances_valid)} validation instances for {valid_size} samples")
    print(f"{len(instances_test)} test instances for {test_size} samples")

    # create output directory; allow reruns on an existing dataset path
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    collect_samples(instances_train, out_dir + '/train', rng, train_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit,
                    dual_top_k=args.dual_top_k)

    rng = np.random.RandomState(args.seed + 1)
    collect_samples(instances_valid, out_dir + '/valid', rng, test_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit,
                    dual_top_k=args.dual_top_k)

    rng = np.random.RandomState(args.seed + 2)
    collect_samples(instances_test, out_dir + '/test', rng, test_size,
                    args.njobs, exploration_policy=exploration_strategy,
                    query_expert_prob=node_record_prob,
                    time_limit=time_limit,
                    dual_top_k=args.dual_top_k)
