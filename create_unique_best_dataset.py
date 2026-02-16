from pathlib import Path
import random
import torch

from utils import load_gzip, GraphDataset

dataset_path = Path('legacy_code_generator/data/samples/setcover/500r_1000c_0.05d')
output_dir = Path('legacy_code_generator/data/samples/setcover/unique_best')
output_dir.mkdir(parents=True, exist_ok=True)
manifest_name = "sample_files.txt"

for split in ["train", "valid", "test"]:
    dataset = GraphDataset(sample_files=[])
    saved = 0
    selected_sample_names = []
    split_dir = dataset_path / split
    output_split_dir = output_dir / split
    output_split_dir.mkdir(parents=True, exist_ok=True)
    for sample_path in sorted(split_dir.glob('sample_*.pkl')):
        sample = load_gzip(sample_path)
        sample_state, _, sample_action, sample_action_set, sample_scores = sample['data']
        _, _, variable_dict = sample_state

        variable_names = variable_dict['names']
        variable_feature_indices = {name: variable_names.index(name) for name in variable_names}
        variable_features = torch.as_tensor(variable_dict['values'], dtype=torch.float32)

        candidates = torch.as_tensor(sample_action_set, dtype=torch.int64)
        candidate_scores = torch.as_tensor(sample_scores, dtype=torch.float32)

        try:
            _, cleaned_scores, _ = dataset.clean_candidates(
            candidates,
            candidate_scores,
            sample_action,
            variable_features,
            variable_feature_indices,
            sample_path.name,
            )
        except ValueError:
            continue
        max_score = cleaned_scores.max()
        if (cleaned_scores == max_score).sum().item() == 1:
            selected_sample_names.append(sample_path.name)
            saved += 1

    manifest_path = output_split_dir / manifest_name
    manifest_content = "\n".join(selected_sample_names)
    manifest_path.write_text(f"{manifest_content}\n" if manifest_content else "", encoding="utf-8")
    print(f'saved {saved} samples to {output_split_dir}')


subset_name = "100overfit"
subset_sizes = {
    "train": 100,
    "valid": 100, 
    "test": 100,
}
subset_seed = 1

subset_output_dir = output_dir.parent / subset_name
subset_output_dir.mkdir(parents=True, exist_ok=True)
rng = random.Random(subset_seed)

for split, requested_size in subset_sizes.items():
    source_split_dir = output_dir / split
    target_split_dir = subset_output_dir / split
    target_split_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source_split_dir / manifest_name
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"missing source manifest for split '{split}': {source_manifest_path}")
    available_sample_names = source_manifest_path.read_text(encoding="utf-8").splitlines()
    if requested_size < 0:
        raise ValueError(f"requested_size must be non-negative for split '{split}', got {requested_size}.")
    if requested_size > len(available_sample_names):
        raise ValueError(
            f"requested_size={requested_size} exceeds available samples ({len(available_sample_names)}) "
            f"for split '{split}'."
        )

    sampled_names = rng.sample(available_sample_names, k=requested_size)
    sampled_names.sort()
    manifest_path = target_split_dir / manifest_name
    manifest_content = "\n".join(sampled_names)
    manifest_path.write_text(f"{manifest_content}\n" if manifest_content else "", encoding="utf-8")
    print(f"saved {len(sampled_names)} sampled file names to {manifest_path}")
