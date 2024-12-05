import torch
from lightning_repertory.surrogate_policy import SurrogatePolicy
from ray.rllib.utils.annotations import override
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns


class Lightning(TorchRLModule):
    @override(TorchRLModule)
    def setup(self):
        if self.model_config.get('use_gpu'):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.lightning_module: SurrogatePolicy = SurrogatePolicy.load_from_checkpoint(self.model_config.get('checkpoint_path'), map_location=torch.device(device))

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        with torch.no_grad():
            action_distribution_inputs = self.lightning_module(batch[Columns.OBS])
        return {
            Columns.ACTION_DIST_INPUTS: action_distribution_inputs,
        }
