from pathlib import Path
from typing import List


def process_surrogate_policy_checkpoint_paths(surrogate_policy_checkpoint_paths: Path):
    all_surrogate_policy_checkpoint_paths = []

    # # for path in surrogate_policy_checkpoint_paths:
    #     path = Path(path)

    if not surrogate_policy_checkpoint_paths.is_absolute():
        path = Path.cwd() / surrogate_policy_checkpoint_paths
    else:
        path = surrogate_policy_checkpoint_paths

    if path.is_dir():
        all_surrogate_policy_checkpoint_paths = all_surrogate_policy_checkpoint_paths + [file.resolve() for file in Path(path).rglob("*.ckpt")]

    elif path.is_file() and path.suffix == '.ckpt':
        print(f"Checkpoint file: {path}")
        all_surrogate_policy_checkpoint_paths.append(path)
    else:
        print(f"Warning: {path} is neither a directory containing .ckpt files nor a valid .ckpt file.")

    print("\nFinal list of checkpoints used:")
    for checkpoint in all_surrogate_policy_checkpoint_paths:
        print(f"  - {checkpoint}")

    return all_surrogate_policy_checkpoint_paths