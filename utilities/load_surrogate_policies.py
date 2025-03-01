from typing import List
from pathlib import Path

from lightning_repertory.surrogate_policy import SurrogatePolicy


def load_surrogate_policies(surrogate_policy_checkpoint_paths: List[Path]) -> List[SurrogatePolicy]:
    surrogate_policies = []

    for surrogate_policy_checkpoint_path in surrogate_policy_checkpoint_paths:
        surrogate_policy: SurrogatePolicy = SurrogatePolicy.load_from_checkpoint(surrogate_policy_checkpoint_path)
        surrogate_policy.eval()
        surrogate_policies.append(surrogate_policy)

    return surrogate_policies